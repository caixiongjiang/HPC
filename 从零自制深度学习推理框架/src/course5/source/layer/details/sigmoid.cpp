//
// Created by 蔡雄江 on 2023/9/5.
//

#include "sigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
    InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
        //判断输入是否为空
        if (inputs.empty()) {
            LOG(ERROR) << "The input tensor array in the sigmoid layer is empty";
            return InferStatus::kInferFailedInputEmpty;
        }

        // 判断输入输出维度是否相同
        if (inputs.size() != outputs.size()) {
            LOG(ERROR) << "The input and output tensor array size of the sigmoid layer do not match";
            return InferStatus::kInferFailedInputOutSizeMatchError;
        }

        const uint32_t batch_size = inputs.size();
        for (uint32_t i = 0; i < batch_size; ++i) {
            const sftensor &input_data = inputs.at(i);
            const sftensor &output_data = outputs.at(i);
            // 判断每一个batch是否为空
            if (input_data == nullptr || input_data->empty()) {
                LOG(ERROR)
                        << "The input tensor array in the sigmoid layer has an empty tensor "
                        << i << " th";
                return InferStatus::kInferFailedInputEmpty;
            }
            //判断每一个batch的维度是否相同
            if (output_data != nullptr && !output_data->empty()) {
                if (input_data->shapes() != output_data->shapes()) {
                    LOG(ERROR) << "The input and output tensor shapes of the sigmoid layer do not match "
                               << i << " th";
                    return InferStatus::kInferFailedInputOutSizeMatchError;
                }
            }
        }

        for (uint32_t i = 0; i < batch_size; ++i) {
            const std::shared_ptr<Tensor<float>> &input = inputs.at(i); // 输入是保持不变的
            CHECK(input != nullptr || !input->empty())
                            << "The input tensor array in the sigmoid layer has an empty tensor "
                            << i << " th";
            std::shared_ptr<Tensor<float>> output = outputs.at(i); // 输出是要改变的，所以是变量
            if (output == nullptr || output->empty()) {
                DLOG(ERROR) << "The output tensor array in the sigmoid layer has an empty tensor "
                            << i << " th";
                output = std::make_shared<Tensor<float>>(input->shapes());
                outputs.at(i) = output;
            }
            CHECK(output->shapes() == input->shapes())
                            << "The input and output tensor shapes of the sigmoid layer do not match "
                            << i << " th";
            /// Sigmoid算子的运算逻辑（取出一个张量的一个数据，进行运算）
            for (uint32_t j = 0; j < input->size(); ++j) {
                float value = input->index(j);
                output->index(j) = 1 / (1.f + expf(-value));
            }
        }
        return InferStatus::kInferSuccess;
    }


    /// 实例化函数
    ParseParameterAttrStatus SigmoidLayer::GetInstance(
            const std::shared_ptr<RuntimeOperator> &op,
            std::shared_ptr<Layer> &sigmoid_layer) {
        CHECK(op != nullptr) << "Sigmoid operator is nullptr";
        sigmoid_layer = std::make_shared<SigmoidLayer>();
        return ParseParameterAttrStatus::kParameterAttrParseSuccess;
    }
    /// 使用工具类注册算子
    LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid", SigmoidLayer::GetInstance);
}