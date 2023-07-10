/**
 * Copyright (c) 2012 MIT License by 6.172 Staff
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/


#include "./util.h"

// Function prototypes
static void merge_a(data_t* A, int p, int q, int r);
static void copy_a(data_t* source, data_t* dest, int n);

// 一个基本的归并排序例程，对子数组 A[p..r]进行排序
void sort_a(data_t* A, int p, int r) {
  assert(A);
  if (p < r) {
    int q = (p + r) / 2;
    sort_a(A, p, q);    //递归地对左半部分排序
    sort_a(A, q + 1, r);//递归地对右半部分排序
    merge_a(A, p, q, r);//合并两个已排序的子数组
  }
}

// 合并例程。合并子数组A[p..q]今儿A[q + 1..r]。
// 在合并操作中使用两个数组 ’left‘ 和 ’right‘。
static void merge_a(data_t* A, int p, int q, int r) {
  assert(A);
  assert(p <= q);
  assert((q + 1) <= r);
  int n1 = q - p + 1;    // 左子数组的大小
  int n2 = r - q;        // 右子数组的大小

  data_t* left = 0, * right = 0;
  mem_alloc(&left, n1 + 1);     // 为左子数组分配内存
  mem_alloc(&right, n2 + 1);    // 为右子数组分配内存
  if (left == NULL || right == NULL) {
    mem_free(&left);
    mem_free(&right);
    return;
  }

  copy_a(&(A[p]), left, n1);       // 将左子数组复制到 left 数组
  copy_a(&(A[q + 1]), right, n2);  // 将右子数组复制到 right 数组
  left[n1] = UINT_MAX;             // 设置左子数组的哨兵元素
  right[n2] = UINT_MAX;            // 设置右子数组的哨兵元素

  int i = 0;
  int j = 0;

  for (int k = p; k <= r; k++) {   // 合并两个子数组
    if (left[i] <= right[j]) {
      A[k] = left[i];
      i++;
    } else {
      A[k] = right[j];
      j++;
    }
  }
  mem_free(&left);                 // 释放左子数组的内存
  mem_free(&right);                // 释放右子数组的内存
}

static void copy_a(data_t* source, data_t* dest, int n) {
  assert(dest);
  assert(source);

  for (int i = 0 ; i < n ; i++) {
    dest[i] = source[i];
  }
}


int main()
{
  const int arraySize = 10;
  data_t dataArray[arraySize];

  for (int i = 0; i < 10; i++) {
    dataArray[i] = 0;
  }

  dataArray[0] = 42;
  dataArray[1] = 100;
  dataArray[2] = 56;
  dataArray[3] = 32;
  dataArray[4] = 54;
  dataArray[5] = 19;
  dataArray[6] = 91;
  dataArray[7] = 20;
  dataArray[8] = 15;
  dataArray[9] = 65;

  sort_a(dataArray, 0, 9);
  // 打印数组中的元素
  for (int i = 0; i < arraySize; i++) {
      printf("%d ", dataArray[i]);
  }
  printf("\n");

  return 0;
}
