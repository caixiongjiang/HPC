// Copyright (c) 2012 MIT License by 6.172 Staff

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Typedefs */

typedef uint32_t data_t;

extern void quickSortIterative(data_t arr[], int l, int h);

/* Insertion sort */
void isort(data_t* left, data_t* right) {
  data_t* cur = left + 1;
  while (cur <= right) {
    data_t val = *cur;
    data_t* index = cur - 1;

    while (index >= left && *index > val) {
      *(index + 1) = *index;
      index--;
    }

    *(index + 1) = val;
    cur++;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Error: wrong number of arguments.\n");
    exit(-1);
  }
  int N = atoi(argv[1]);
  int K = atoi(argv[2]);
  unsigned int seed = 42;
  printf("Sorting %d values...\n", N);
  data_t* data = (data_t*) malloc(N * sizeof(data_t));
  if (data == NULL) {
    free(data);
    printf("Error: not enough memory\n");
    exit(-1);
  }

  int i, j;
  for (j = 0; j < K ; j++) {
    for (i = 0; i < N; i++) {
      data[i] = rand_r(&seed);
      //  printf("%d ", data[i]);
    }
    //  printf("\n");

    isort(data, data + N - 1);
    //quickSortIterative(data, 0, N);
    /*for (i = 0; i < N; i++) {
      printf("%d ", data[i]);
    }
    printf("\n");*/
  }
  free(data);
  printf("Done!\n");
  return 0;
}
