//
// Created by cxj on 2023-08-15.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

/// 需要自己完成的是：
/// Tensor::Flatten函数 : 将多维数据展开变成一维
/// Tensor::Padding函数 : 深度学习中的padding，接受4个参数，分别代表上下左右的填充维度，以及接受一个填充数值的参数
/// :: 代表作用域分解运算符
/// 声明了一个类A，类A里声明了一个成员函数voidf()，但没有在类的声明里给出f的定义，
/// 那么在类外定义f时，就要写成voidA::f()，表示这个f()函数是类A的成员函数。
namespace kuiper_infer {
  /* 创建三维张量 */
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
  // 当channels和rows同时等于1时，raw_shapes的长度也会是1，表示此时的Tensor是一维的
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    // 当channels等于1时，raw_shapes的长度也会是2，表示此时的Tensor是二维的
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

/* 创建一维张量 */
Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

/* 创建二维张量 */
Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);

  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

void Tensor<float>::set_data(const arma::fcube& data) {
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const { return this->data_.empty(); }

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::fcube& Tensor<float>::data() { return this->data_; }

const arma::fcube& Tensor<float>::data() const { return this->data_; }

arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  // 四周填充的维度
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  /// Homework2：请补充代码
  const uint32_t channels = this->channels();
  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();

  const uint32_t new_rows = rows + pad_rows1 + pad_rows2;
  const uint32_t new_cols = cols + pad_cols1 + pad_cols2;
  auto new_tensor = arma::fcube(new_rows, new_cols, channels);
  new_tensor.fill(padding_value);

  /// `cols()` 和 `rows()` 方法分别返回数据张量的列数和行数。
  /// 这个张量的大小是在创建对象时指定的，一旦创建后就不能再改变大小。
  /// 拷贝原来张量中的元素到新的张量中
  for (uint32_t i = 0; i < channels; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
          for (uint32_t k = 0; k < rows; ++k) {
              //计算新的单元位置
              uint32_t new_j = j + pad_cols1;
              uint32_t new_k = k + pad_rows1;
              //拷贝元素
              new_tensor.at(new_k, new_j, i) = this->data_.at(k, j, i);
          }
      }
  }
  // 将新的张量赋值给 data_ 使用move函数，直接将整个对象拷贝了过来
  this->data_ =  std::move(new_tensor);
  // 我们在创建new_tensor的时候已经定义了cols，rows，channels。 还需要指定类内成员raw_shape()_的值
  this->raw_shapes_ = std::vector<uint32_t>{channels, new_rows, new_cols};
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->data_.n_slices;

    for (uint32_t i = 0; i < channels; ++i) {
      auto& channel_data = this->data_.slice(i);
      const arma::fmat& channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());
      channel_data = channel_data_t.t();
    }
  } else {
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());
  /// Homework1: 请补充代码
  const uint32_t total_elems = this->data_.size();
  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t channels = this->data_.n_slices;

  std::vector<float> flattened_data(total_elems);

  if (row_major) {
      // 按行主序取数据, arma::fcube数据中，每个channel内的取值顺序是按列主序的，所以需要改变一下取值顺序
      for (uint32_t i = 0; i < channels; ++i) {
          auto& channel_data = this->data_.slice(i);
          // t()代表转置，&的作用：channel_data.t()` 返回的是一个新的 `arma::fmat` 对象，
          // 如果没有使用引用类型的变量来接收它，那么程序就无法操作这个新的对象。
          const arma::fmat &channel_data_t = channel_data.t();
          std::copy(channel_data_t.begin(), channel_data_t.end(), flattened_data.begin() + i * rows * cols);
      }
  } else {
      /// memptr()属于arma数学库中数据的方法，而begin()是c++中vector的方法，作用是相同的，都是返回第一个元素的地址
      // 因为arma的数据本身就是列主序，只要将数据所有拷贝到flatten_data中就好了
      std::copy(this->data_.memptr(), this->data_.memptr() + total_elems, flattened_data.begin());
  }

  // 改变数组的shapes
  this->data_.set_size(1, total_elems, 1);
  // 将flatten数据拷贝回原来的张量，第三个参数接收的是需要拷贝的起始地址
  std::copy(flattened_data.begin(), flattened_data.end(), this->data_.memptr());
  this->raw_shapes_ = std::vector<uint32_t>{total_elems};
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  // 调用内部类
  this->data_.transform(filter);
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  CHECK_LE(this->raw_shapes_.size(), 3);
  CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  //accumulate函数主要用于累加，前两个参数用于指定范围，第三个参数代表初始值
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  std::vector<float> values;
  if (row_major) {
    values = this->values(true);
  }
  //调用arma:cube类内部的reshape方法 参数顺序（row, col, channel）
  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    this->data_.reshape(1, shapes.at(0), 1);
    this->raw_shapes_ = {shapes.at(0)};
  }

  if (row_major) {
    this->Fill(values, true);
  }
}

float* Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

float* Tensor<float>::raw_ptr(uint32_t offset) {
  const uint32_t size = this->size();
  CHECK(!this->data_.empty());
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;
}

std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);
  std::vector<float> values(this->data_.size());

  if (!row_major) {
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      const arma::fmat& channel = this->data_.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}

float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  uint32_t offset = index * this->rows() * this->cols();
  CHECK_LE(offset, this->size());
  float* mem_ptr = this->raw_ptr() + offset;
  return mem_ptr;
}
}  // namespace kuiper_infer
