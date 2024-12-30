#include "ModelBuilder.h"

#include <stdio.h>
#include <stdlib.h>

void extractCategories(float inputData[4000][3], float categories[4000][4]) {
  for (int i = 0; i < 4000; ++i) {
    int category = (int)inputData[i][2];
    if (category == 1) {
      categories[i][0] = 1;
    } else if (category == 2) {
      categories[i][1] = 1;
    } else if (category == 3) {
      categories[i][2] = 1;
    } else if (category == 4) {
      categories[i][3] = 1;
    } else {
	    printf("Invalid Category");
      exit(-1);
    }
  }
}