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


/**
 * Matrix Multiply
 *
 *
 * Declarations here are your API specification -- we expect that your
 * code defines and correctly implements these functions and does not modify
 * the way the reference implementation uses variables mentioned here,
 * so that we may call them from testbed.c and any other testing software
 * that we write!
 *
 * Deviating from this API may cause your program to fail all of our tests.
 **/

#ifndef MATRIX_MULTIPLY_H_INCLUDED

#define MATRIX_MULTIPLY_H_INCLUDED

typedef struct {
  int rows;
  int cols;
  int** values;
} matrix;

// Multiply matrix A*B, store result in C.
int matrix_multiply_run(const matrix* A, const matrix* B, matrix* C);

// Allocates a row-by-cols matrix and returns it
matrix* make_matrix(int rows, int cols);

// Frees an allocated matrix
void free_matrix(matrix* m);

// Print matrix
void print_matrix(const matrix* m);
#endif  // MATRIX_MULTIPLY_H_INCLUDED
