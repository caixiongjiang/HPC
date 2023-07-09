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


#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>

#include "./fasttime.h"
#include "./tests.h"

// Call TEST_PASS() from your test cases to mark a test as successful
#define TEST_PASS() TEST_PASS_WITH_NAME(__func__, __LINE__)

#define TEST_PASS_WITH_NAME(name, line) \
    fprintf(stderr, " --> %s at line %d: PASS\n", (name), (line))

// Call TEST_FAIL from your test cases to mark a test as failed. TEST_FAIL
// should print a meaningful message with the reason that the test failed.
//
// Calling it is just like calling printf().
#define TEST_FAIL(failure_msg, args...) \
    TEST_FAIL_WITH_NAME(__func__, __LINE__, failure_msg, ##args)

#define TEST_FAIL_WITH_NAME(name, line, failure_msg, args...) do { \
    fprintf(stderr, " --> %s at line %d: FAIL\n    Reason:", (name), (line)); \
    fprintf(stderr, (failure_msg), ## args); \
    fprintf(stderr, "\n"); \
} while (0)




/* Some global variables to make it easier to run individual tests. */
// static int test_verbose = 1 ;
static unsigned int randomSeed = 1;

static inline void display_array(data_t* data, int N) {
  // display array
  for (int i = 0; i < N; i++) {
    printf("%d ", data[i]);
  }
  printf("\n");
}

static inline void copy_data(data_t* data, data_t* data_bcup, int N) {
  // copy data_bcup to data
  for (int i = 0 ; i < N ; i++) {
    *data++ = *data_bcup++;
  }
}

static inline int post_process(data_t* data, data_t* data_bcup, int N,
                               int printFlag, char* name, int begin, int end) {
  int result = 1;
  if (printFlag) {
    printf("%s: ", name);
    printf("Data after sort\n");
    display_array(data, N);
  }

  // check if the array is unchanged from data[0..begin-1]
  for (int i = 0 ; i < begin ; i++) {
    if (data[i] != data_bcup[i]) {
      printf("%s: ", name);
      TEST_FAIL("Array outside sort boundary changed!\n");
      result = 0;
      break;
    }
  }

  // check if sub-array is sorted
  for (int i = begin + 1 ; i < end + 1 ; i++) {
    if (data[i - 1] > data[i]) {
      printf("%s: ", name);
      TEST_FAIL("Arrays are sorted: NO!\n");
      result = 0;
      break;
    }
  }

  // check if the array is unchanged from data[end+1..N-1]
  for (int i = end + 1 ; i < N ; i++) {
    if (data[i] != data_bcup[i]) {
      printf("%s: ", name);
      TEST_FAIL("Array outside sort boundary changed!\n");
      result = 0;
      break;
    }
  }
  copy_data(data, data_bcup, N);
  return result;
}

static
void
init_data(data_t* data, int N, int randomPrefix, int invertedSuffix) {
  // initialize data with randomPrefix random numbers
  assert(randomPrefix <= N);
  /* random prefix */
  for (int i = 0; i < randomPrefix; i++) {
    data[i] = rand_r(&randomSeed) % RANGE;
  }
  /* sorted or inverted suffix - duplicates are OK */
  for (int i = randomPrefix; i < N; i++) {
    if (invertedSuffix) {
      data[i] = (N - i) % RANGE;         // inverted
    } else {
      data[i] = i % RANGE;             // sorted array
    }
  }
}


struct dataGenerator_t {
  void (*generate)(data_t*, int);
  char* name;
};

void all_random(data_t* data, int N) {
  init_data(data, N, N, 0);
}

void all_inverted(data_t* data, int N) {
  init_data(data, N, 0, 1 /* inverted */);
}

struct dataGenerator_t dataGen[] = {
  {all_random, "random"},
  {all_inverted, "inverted"},
};

static void test_correctness(int printFlag, int N, int R,
                             struct testFunc_t* testFunc, int numFunc) {
  fasttime_t time1, time2;
  data_t* data, *data_bcup;
  int success = 1;

  float* sum_time = (float*) alloca(numFunc * sizeof(float));
  for (int i = 0; i < numFunc; i++) {
    sum_time[i] = 0;
  }

  // allocate memory
  data = (data_t*) malloc(N * sizeof(data_t));
  data_bcup = (data_t*) malloc(N * sizeof(data_t));

  if (data == NULL || data_bcup == NULL) {
    printf("Error: not enough memory\n");
    free(data);
    free(data_bcup);
    exit(-1);
  }

  // Initialize elapsed time counters to 0.
  for (int k = 0; k < numFunc; k++) {
    sum_time[k] = 0;
  }

  // repeat for each array kind
  for (int gen = 0;
       gen < sizeof(dataGen) / sizeof(dataGen[0]);
       gen++) {
    printf("Generating %s array of %d elements\n",
           dataGen[gen].name, N);
    // repeat for each trial
    for (int j = 0; j < R; j++) {
      // generate new data for each trial
      dataGen[gen].generate(data, N);

      if (printFlag) {
        printf("Data before sort\n");
        display_array(data, N);
      }
      for (int i = 0; i < N; i++) {
        data_bcup[i] = data[i];
      }

      for (int k = 0; k < numFunc; k++) {
        time1 = gettime();
        testFunc[k].func(data, 0, N - 1);
        time2 = gettime();

        sum_time[k] += tdiff(time1, time2);
        success &= post_process(data, data_bcup, N, printFlag, testFunc[k].name, 0, N - 1);

        if (!success) {
          break;
        }
      }
    }

    if (success) {
      printf("Arrays are sorted: yes\n");
      TEST_PASS();
      // report average execution time over R runs.
      for (int k = 0; k < numFunc; k++) {
        float avgRuntime = R > 0 ? sum_time[k] / R : 0;
        printf("%s: Elapsed execution time: %f sec\n", testFunc[k].name, avgRuntime);
      }
    }
  }

  free(data);
  free(data_bcup);
  return;
}

static void test_zero_element(int printFlag, int N, int R,
                              struct testFunc_t* testFunc, int numFunc) {
  int success = 1;
  for (int i = 0; i < numFunc; i++) {
    data_t data[] = {0, 0, 0};
    testFunc[i].func(&data[1], 0, 0);
    if (data[0] != 0 && data[1] != 0 && data[2] != 0) {
      printf("Error: %s failed to sort array with zero element\n",
             testFunc[i].name);
      success = 0;
    }
  }

  if (success) {
    TEST_PASS();
  } else {
    TEST_FAIL("Sorting array with zero element failed");
  }
}

static void test_one_element(int printFlag, int N, int R,
                             struct testFunc_t* testFunc, int numFunc) {
  int success = 1;
  for (int i = 0; i < numFunc; i++) {
    data_t data[] = {0, 1, 0};
    testFunc[i].func(&data[1], 0, 0);
    if (data[0] != 0 && data[2] != 0) {
      // Can't test data[1] == 1 because of final part of homework.
      printf("Error: %s failed to sort array with one element\n",
             testFunc[i].name);
      success = 0;
    }
  }

  if (success) {
    TEST_PASS();
  } else {
    TEST_FAIL("Sorting array with one element failed");
  }
}

void test_subarray(int printFlag, int N, int R,
                   struct testFunc_t* testFunc, int numFunc) {
  data_t* data, *data_bcup;
  int success = 1;

  // allocate memory
  data = (data_t*) malloc(N * sizeof(data_t));
  data_bcup = (data_t*) malloc(N * sizeof(data_t));

  if (data == NULL || data_bcup == NULL) {
    printf("Error: not enough memory\n");
    free(data);
    free(data_bcup);
    exit(-1);
  }

  // initialize data with random numbers
  for (int i = 0; i < N; i++) {
    data[i] = rand_r(&randomSeed);
    data_bcup[i] = data[i];
  }
  if (printFlag) {
    printf("Data before sort\n");
    display_array(data, N);
  }
  int begin = rand_r(&randomSeed) % N;
  int end = N - 1 - begin;
  if (begin > end) {
    int temp = begin;
    begin = end;
    end = temp;
  }

  printf("Sorting subarray A[%d..%d]\n", begin, end);
  for (int i = 0; i < numFunc; i++) {
    testFunc[i].func(data, begin, end);
    success &= post_process(data, data_bcup, N, printFlag, testFunc[i].name, begin, end);
  }

  if (success) {
    printf("Arrays are sorted: yes\n");
    TEST_PASS();
  } else {
    TEST_FAIL("Sorting subarray failed");
  }

  free(data);
  free(data_bcup);
  return;
}

test_case test_cases[] = {
  test_correctness,
  test_zero_element,
  test_one_element,
  // test_subarray,
  // ADD YOUR TEST CASES HERE
  NULL  // This marks the end of all test cases. Don't change this!
};
