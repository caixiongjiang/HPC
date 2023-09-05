//
// Created by 蔡雄江 on 2023/9/5.
//

#ifndef KUIPER_DATAWHALE_SIGMOID_HPP
#define KUIPER_DATAWHALE_SIGMOID_HPP

#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer{

class SigmoidLayer : public NonParamLayer{
public:
    SigmoidLayer() : NonParamLayer("Sigmoid"){}
        InferStatus Forward(
            const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
            std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;
        static ParseParameterAttrStatus GetInstance(
            const std::shared_ptr<RuntimeOperator>& op,
            std::shared_ptr<Layer>& sigmoid_layer);
};
} // namespace kuiper_infer
#endif //KUIPER_DATAWHALE_SIGMOID_HPP
