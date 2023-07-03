// Copyright (c) 2012 MIT License by 6.172 Staff

// All occurences of ONE will be replaced by 1.
#define ONE 1

// Macros can also behave similar to inline functions.
// Note that parentheses around arguments are required to preserve order of
// operations. Otherwise, you can introduce bugs when substitution happens

#define MIN(a, b) ((a) < (b) ? (a) : (b))

int c = ONE, d = ONE + 5;
int e = MIN(c, d);

#ifndef NDEBUG
// This code will be compiled only when
// the macro NDEBUG is not defined.
// Recall that if clang is passed -DNDEBUG on the command line,
// then NDEBUG will be defined.
  if (something) {}
#endif
