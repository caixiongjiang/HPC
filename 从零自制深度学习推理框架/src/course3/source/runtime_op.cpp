//
// Created by cxj on 2023-08-21.
//
#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
RuntimeOperator::~RuntimeOperator() {
  for (auto& [_, param] : this->params) {
    if (param != nullptr) {
      delete param;
      param = nullptr;
    }
  }
}

}  // namespace kuiper_infer
