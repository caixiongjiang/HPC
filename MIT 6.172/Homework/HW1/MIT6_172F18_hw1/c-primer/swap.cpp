
#include <stdio.h>
#include <iostream>

void swap(int &i, int &j) {
  int temp = i;
  i = j;
  j = temp;
}

int main() {
  int k = 1;
  int m = 2;
  swap(k, m);
  printf("k = %d, m = %d\n", k, m);

  return 0;
}