//
// Created by fss on 23-7-22.
//
#include "layer/abstract/layer_factory.hpp"
#include "../source/layer/details/convolution.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace kuiper_infer;

/// 只测试了group为1的情况
TEST(test_registry, create_layer_convforward) {
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs(batch_size);

  const uint32_t in_channel = 2;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
    input->data().slice(0) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";

    input->data().slice(1) = "1,2,3,4;"
                             "5,6,7,8;"
                             "9,10,11,12;"
                             "13,14,15,16;";
    inputs.at(i) = input;
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 2;
  /// 输入：1，2，4，4（BCHW）-->3*3 Conv，stride=1，channel（卷积核个数）=2 --> 输出：1，2，2，2
  std::vector<sftensor> weights;
  /// 自定义卷积核的权重
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->data().slice(0) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    kernel->data().slice(1) = arma::fmat("1,2,3;"
                                         "3,2,1;"
                                         "1,2,3;");
    weights.push_back(kernel);
  }
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs);
  outputs.at(0)->Show();
}

/// group为2的情况 （通道数和卷积核个数必须为偶数）
TEST(test_registry, create_layer_convforward_group) {
    const uint32_t batch_size = 1;
    std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
    std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);

    const uint32_t in_channel = 2;
    for (uint32_t i = 0; i < batch_size; ++i) {
        sftensor input = std::make_shared<ftensor>(in_channel, 4, 4);
        input->data().slice(0) = arma::fmat("1,2,3,4;"
                                            "5,6,7,8;"
                                            "9,10,11,12;"
                                            "13,14,15,16;");

        input->data().slice(1) = arma::fmat("1,2,3,4;"
                                            "5,6,7,8;"
                                            "9,10,11,12;"
                                            "13,14,15,16;");
        inputs.at(i) = input;
    }

    const uint32_t kernel_h = 3;
    const uint32_t kernel_w = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t kernel_count = 2;
    const uint32_t group = 2;  // 设置分组数为2

    std::vector<sftensor> weights;
    for (uint32_t i = 0; i < kernel_count; ++i) {
        /// channel数计算： in_channel / group  kernel的计算次数也要随着分组减少
        sftensor kernel = std::make_shared<Tensor<float>>(in_channel / group, kernel_h, kernel_w);  // 更新卷积核的通道数
        kernel->data().slice(0) = arma::fmat("1,2,3;"
                                             "3,2,1;"
                                             "1,2,3;");
        weights.push_back(kernel);
    }

    ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                                0, stride_h, stride_w, group, false);  // 设置分组数
    conv_layer.set_weights(weights);
    conv_layer.Forward(inputs, outputs);

    outputs.at(0)->Show();
}