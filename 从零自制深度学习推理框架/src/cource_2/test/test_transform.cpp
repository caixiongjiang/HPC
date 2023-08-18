//
// Created by cxj on 2023-08-17.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

/* 定义每个元素都减去1的transform功能 */
float MinusOne(float value) { return value - 1.f; }
TEST(test_transform, transform1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  f1.Transform(MinusOne);
  f1.Show();
}

float AddOne(float value) { return value + 1.f; }
TEST(test_transform, transform2) {
    using namespace kuiper_infer;
    Tensor<float> f2(2, 6, 7);
    f2.Rand();
    f2.Show();
    f2.Transform(AddOne);
    f2.Show();
}