// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-13.

#include "convolution.hpp"
#include <glog/logging.h>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
ConvolutionLayer::ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                                   uint32_t kernel_h, uint32_t kernel_w,
                                   uint32_t padding_h, uint32_t padding_w,
                                   uint32_t stride_h, uint32_t stride_w,
                                   uint32_t groups, bool use_bias)
    : ParamLayer("Convolution"),
      use_bias_(use_bias),
      groups_(groups),
      padding_h_(padding_h),
      padding_w_(padding_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {
  if (groups != 1) {
    in_channel /= groups;
  }
  this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
  if (use_bias_) {
    this->InitBiasParam(output_channel, 1, 1, 1);
  }
}

InferStatus ConvolutionLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the convolution layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the convolution "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (weights_.empty()) {
    LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                  "be greater than zero";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
    LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
    return InferStatus::kInferFailedBiasParameterError;
  }

  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }

  const uint32_t kernel_count = this->weights_.size();
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t row_len = kernel_h * kernel_w;
  CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
      << "The size of kernel matrix in the convolution layer should be greater "
         "than zero";

  for (uint32_t k = 0; k < kernel_count; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_c);
  }
  const uint32_t kernel_count_group = kernel_count / groups_;
  const uint32_t batch_size = inputs.size();

  if (kernel_matrix_arr_.empty()) {
    this->InitIm2ColWeight();
  }

  if (!kernel_matrix_arr_.empty()) {
    if (groups_ == 1) {
      CHECK(kernel_matrix_arr_.size() == kernel_count_group)
          << "The number of kernel matrix and kernel_count_group do not match";
    } else {
      CHECK(kernel_matrix_arr_.size() == kernel_count)
          << "The number of kernel matrix and kernel_count do not match";
    }
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the convolution layer has an empty  "
           "tensor "
        << i << " th";

    const uint32_t input_c = input->channels();
    const uint32_t input_padded_h = input->rows() + 2 * padding_h_;
    const uint32_t input_padded_w = input->cols() + 2 * padding_w_;

    const uint32_t output_h =
        std::floor((int(input_padded_h) - int(kernel_h)) / stride_h_ + 1);
    const uint32_t output_w =
        std::floor((int(input_padded_w) - int(kernel_w)) / stride_w_ + 1);
    CHECK(output_h > 0 && output_w > 0)
        << "The size of the output tensor should be greater than zero " << i
        << " th";

    if (groups_ != 1) {
      CHECK(kernel_count % groups_ == 0);
      CHECK(input_c % groups_ == 0);
    }

    uint32_t col_len = output_h * output_w;
    CHECK(col_len > 0) << "Output_h x output_w for the convolution layer "
                          "should be greater than zero "
                       << i << " th";

    uint32_t input_c_group = input_c / groups_;
    CHECK(input_c_group == kernel_c) << "The number of channel for the kernel "
                                        "matrix and input tensor do not match";

    for (uint32_t g = 0; g < groups_; ++g) {
      const auto& input_matrix =
          Im2Col(input, kernel_w, kernel_h, input->cols(), input->rows(),
                 input_c_group, g, row_len, col_len);
      std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
      if (output_tensor == nullptr || output_tensor->empty()) {
        output_tensor =
            std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
        outputs.at(i) = output_tensor;
      }

      CHECK(output_tensor->rows() == output_h &&
            output_tensor->cols() == output_w &&
            output_tensor->channels() == kernel_count)
          << "The output tensor array in the convolution layer has an "
             "incorrectly sized tensor "
          << i << "th";

      const uint32_t kernel_count_group_start = kernel_count_group * g;
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        arma::frowvec kernel;
        if (groups_ == 1) {
          kernel = kernel_matrix_arr_.at(k);
        } else {
          kernel = kernel_matrix_arr_.at(kernel_count_group_start + k);
        }
        ConvGemmBias(input_matrix, output_tensor, g, k, kernel_count_group,
                     kernel, output_w, output_h);
      }
    }
  }
  return InferStatus::kInferSuccess;
}

arma::fmat ConvolutionLayer::Im2Col(sftensor input, uint32_t kernel_w,
                                    uint32_t kernel_h, uint32_t input_w,
                                    uint32_t input_h, uint32_t input_c_group,
                                    uint32_t group, uint32_t row_len,
                                    uint32_t col_len) const {
  arma::fmat input_matrix(input_c_group * row_len, col_len);
  const uint32_t input_padded_h = input_h + 2 * padding_h_;
  const uint32_t input_padded_w = input_w + 2 * padding_w_;
  const float padding_value = 0.f;
  for (uint32_t ic = 0; ic < input_c_group; ++ic) {
    float* input_channel_ptr =
        input->matrix_raw_ptr(ic + group * input_c_group);
    uint32_t current_col = 0;
    uint32_t channel_row = ic * row_len;
    for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += stride_w_) {
      for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += stride_h_) {
        float* input_matrix_ptr =
            input_matrix.colptr(current_col) + channel_row;
        current_col += 1;
        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
          const uint32_t region_w = input_h * (w + kw - padding_w_);
          for (uint32_t kh = 0; kh < kernel_h; ++kh) {
            if ((kh + r >= padding_h_ && kw + w >= padding_w_) &&
                (kh + r < input_h + padding_h_ &&
                 kw + w < input_w + padding_w_)) {
              float* region_ptr =
                  input_channel_ptr + region_w + (r + kh - padding_h_);
              *input_matrix_ptr = *region_ptr;
            } else {
              *input_matrix_ptr = padding_value;  // only support zero mode
            }
            input_matrix_ptr += 1;
          }
        }
      }
    }
  }
  return input_matrix;
}

void ConvolutionLayer::ConvGemmBias(
    const arma::fmat& input_matrix, sftensor output_tensor, uint32_t group,
    uint32_t kernel_index, uint32_t kernel_count_group,
    const arma::frowvec& kernel, uint32_t output_w, uint32_t output_h) const {
  arma::fmat output(
      output_tensor->matrix_raw_ptr(kernel_index + group * kernel_count_group),
      output_h, output_w, false, true);

  CHECK(output.size() == output_h * output_w)
      << "Output_h x output_w for the convolution layer "
         "should be output tensor size";

  if (!this->bias_.empty() && this->use_bias_) {
    std::shared_ptr<Tensor<float>> bias;
    bias = this->bias_.at(kernel_index);
    if (bias != nullptr && !bias->empty()) {
      float bias_value = bias->index(0);
      output = kernel * input_matrix + bias_value;
    } else {
      LOG(FATAL) << "Bias tensor is empty or nullptr";
    }
  } else {
    output = kernel * input_matrix;
  }
}

void ConvolutionLayer::InitIm2ColWeight() {
  const uint32_t kernel_count = this->weights_.size();
  CHECK(kernel_count > 0) << "kernel count must greater than zero";
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t row_len = kernel_h * kernel_w;
  CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
      << "The size of kernel matrix should be greater than zero";

  for (uint32_t k = 0; k < kernel_count; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_c);
  }

  if (groups_ == 1) {
    const uint32_t kernel_count_group = kernel_count / groups_;
    std::vector<arma::frowvec> kernel_matrix_arr(kernel_count_group);
    arma::frowvec kernel_matrix_c(row_len * kernel_c);
    for (uint32_t k = 0; k < kernel_count_group; ++k) {
      const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
      for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
        memcpy(kernel_matrix_c.memptr() + row_len * ic,
               kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
      }
      kernel_matrix_arr.at(k) = kernel_matrix_c;
    }
    this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
  } else {
    // group != 1
    const uint32_t kernel_count_group = kernel_count / groups_;
    std::vector<arma::frowvec> kernel_matrix_arr;
    for (uint32_t g = 0; g < groups_; ++g) {
      arma::fmat kernel_matrix_c(1, row_len * kernel_c);
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        const std::shared_ptr<Tensor<float>>& kernel =
            this->weights_.at(k + g * kernel_count_group);
        for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
          memcpy(kernel_matrix_c.memptr() + row_len * ic,
                 kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
        }
        kernel_matrix_arr.emplace_back(kernel_matrix_c);
      }
    }
    CHECK(kernel_matrix_arr.size() == kernel_count);
    this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
  }
}

ParseParameterAttrStatus ConvolutionLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& conv_layer) {
  CHECK(op != nullptr) << "Convolution operator is nullptr";
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

  if (params.find("dilation") == params.end()) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return ParseParameterAttrStatus::kParameterMissingDilation;
  }

  auto dilation_param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("dilation"));

  if (dilation_param == nullptr || dilation_param->value.size() != 2) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return ParseParameterAttrStatus::kParameterMissingDilation;
  }

  CHECK(dilation_param->value.at(0) != 1 || dilation_param->value.at(1))
      << "Only support dilation value equals to one!";

  if (params.find("in_channels") == params.end()) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }
  auto in_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_channels"));
  if (!in_channel) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }

  if (params.find("out_channels") == params.end()) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  auto out_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_channels"));
  if (!out_channel) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  auto padding =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  auto use_bias =
      std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
  if (!use_bias) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }
  auto stride =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  auto kernel = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("kernel_size"));
  if (!kernel) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  if (params.find("padding_mode") != params.end()) {
    auto padding_mode = std::dynamic_pointer_cast<RuntimeParameterString>(
        params.at("padding_mode"));
    if (padding_mode == nullptr) {
      LOG(ERROR) << "Can not find the padding parameter";
      return ParseParameterAttrStatus::kParameterMissingPaddingMode;
    } else {
      const std::string& padding_mode_str = padding_mode->value;
      if (padding_mode_str != "zeros") {
        LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
        return ParseParameterAttrStatus::kParameterMissingPaddingMode;
      }
    }
  } else {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPaddingMode;
  }

  auto groups =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("groups"));
  if (!groups) {
    LOG(ERROR) << "Can not find the groups parameter";
    return ParseParameterAttrStatus::kParameterMissingGroups;
  }

  const uint32_t dims = 2;
  const std::vector<int>& kernels = kernel->value;
  const std::vector<int>& paddings = padding->value;
  const std::vector<int>& strides = stride->value;
  if (paddings.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (strides.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (kernels.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  // kernel的方向是倒置的
  conv_layer = std::make_shared<ConvolutionLayer>(
      out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
      paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
      groups->value, use_bias->value);

  // load weights
  const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attrs =
      op->attribute;
  if (use_bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Can not find the bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto& bias = attrs.at("bias");
    const std::vector<int>& bias_shape = bias->shape;
    if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
      LOG(ERROR) << "The attribute of bias shape is wrong";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }

    const std::vector<float>& bias_values = bias->get<float>();
    conv_layer->set_bias(bias_values);
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the weight attribute";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const auto& weight = attrs.at("weight");
  const std::vector<int>& weight_shape = weight->shape;
  if (weight_shape.empty()) {
    LOG(ERROR) << "The attribute of weight shape is wrong";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const std::vector<float>& weight_values = weight->get<float>();
  conv_layer->set_weights(weight_values);

  auto conv_layer_derived =
      std::dynamic_pointer_cast<ConvolutionLayer>(conv_layer);
  CHECK(conv_layer_derived != nullptr);
  conv_layer_derived->InitIm2ColWeight();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kConvGetInstance("nn.Conv2d",
                                        ConvolutionLayer::GetInstance);
}  // namespace kuiper_infer
