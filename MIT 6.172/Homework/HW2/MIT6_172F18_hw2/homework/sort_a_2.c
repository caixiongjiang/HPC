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

data_t* tmp = NULL;  // 声明全局变量 tmp

void merge_sort(data_t* arr, int n) {
  assert(arr);

  tmp = (data_t*) malloc(n * sizeof(data_t));
  if (tmp == NULL) {
    printf("Error: not enough memory\n");
    return;
  }

  // 接下来的归并排序代码保持不变
  for (int size = 1; size < n; size *= 2) {
    for (int left = 0; left < n - size; left += 2 * size) {
      int mid = left + size - 1;
      int right = left + 2 * size - 1;
      if (right >= n) {
        right = n - 1;
      }
      merge(arr, left, mid, right);
    }
  }
}

void merge(data_t* arr, int left, int mid, int right) {
  // 归并操作中的临时数组变量 tmp 不再声明
  int i = left;
  int j = mid + 1;
  int k = left;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      tmp[k++] = arr[i++];
    } else {
      tmp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    tmp[k++] = arr[i++];
  }

  while (j <= right) {
    tmp[k++] = arr[j++];
  }

  for (int p = left; p <= right; p++) {
    arr[p] = tmp[p];
  }
}
