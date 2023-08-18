//
// Created by cxj on 2023-08-17.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(test_tensor_values, tensor_values1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  // slice方法在通道层面上获取矩阵
  LOG(INFO) << "Data in the first channel: " << f1.slice(0);
  // at方法则是获取某一个数
  LOG(INFO) << "Data in the (1,1,1): " << f1.at(1, 1, 1);
}