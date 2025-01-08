#include <stdlib.h>
#include <stdio.h>
#include "DataGenerator.h"

/**
 * Returns the category in which point x1, x2 belongs to.
 * x1 float
 * x2 float
 * returns int
 */
int getClassificationCategory(float x1, float x2) {
	if ((x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2) {
		if (x2 > 0.5) {
			return 1;
		} else if (x2 < 0.5) {
			return 2;
		}
	} else if ((x1 + 0.5) * (x1 + 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2) {
		if (x2 > -0.5) {
			return 1;
		} else if (x2 < -0.5) {
			return 2;
		}
	} else if ((x1 - 0.5) * (x1 - 0.5) + (x2 + 0.5) * (x2 + 0.5) < 0.2) {
		if (x2 > -0.5) {
			return 1;
		} else if (x2 < -0.5) {
			return 2;
		}
	} else if ((x1 + 0.5) * (x1 + 0.5) + (x2 - 0.5) * (x2 - 0.5) < 0.2) {
		if (x2 > 0.5) {
			return 1;
		} else if (x2 < 0.5) {
			return 2;
		}
	} else {
		if (x1 * x2 > 0) {
			return 3;
		} else if (x1 * x2 < 0) {
			return 4;
		}
	}
}

void createClassificationTrainData(float data[4000][3]) {
	for (int i = 0; i < 4000; ++i) {
		// float x1 = -1 + 2 * ((float)rand()/RAND_MAX);
		float x1 = generateRandomFloat(-1.0, 1.0);
		// float x2 = -1 + 2 * ((float)rand()/RAND_MAX);
		float x2 = generateRandomFloat(-1.0, 1.0);
		float category = (float) getClassificationCategory(x1, x2);
		data[i][0] = x1;
		data[i][1] = x2;
		data[i][2] = category;
	}
}

void createClassificationTestData(float data[4000][4]) {
	for (int i = 0; i < 4000; ++i) {
		// float x1 = -1 + 2 * ((float)rand()/RAND_MAX);
		float x1 = generateRandomFloat(-1.0, 1.0);
		// float x2 = -1 + 2 * ((float)rand()/RAND_MAX);
		float x2 = generateRandomFloat(-1.0, 1.0);
		float category = (float) getClassificationCategory(x1, x2);
		data[i][0] = x1;
		data[i][1] = x2;
		data[i][2] = category;
		data[i][3] = 0.0;
	}
}

void exportClassificationTrainData(float data[4000][3], char* title) {
  FILE *file = fopen(title, "w");
  if (file == NULL) {
    printf("Error opening file\n");
    return;
  }

  for (int i = 0; i < 4000; ++i) {
    fprintf(file, "%f %f %d\n", data[i][0], data[i][1], (int)(data[i][2]));
  }
}

void exportClassificationTestData(float data[4000][4], char* title) {
  FILE *file = fopen(title, "w");
  if (file == NULL) {
    printf("Error opening file\n");
    return;
  }

  for (int i = 0; i < 4000; ++i) {
    fprintf(file, "%f %f %d\n", data[i][0], data[i][1], (int)(data[i][2]));
  }
}

void exportTestData(float data[4000][4], char* title) {
  FILE *file = fopen(title, "w");
  if (file == NULL) {
    printf("Error opening file\n");
    return;
  }

  for (int i = 0; i < 4000; ++i) {
    fprintf(file, "%f %f %d %d\n", data[i][0], data[i][1], (int)(data[i][2]),
      (int)(data[i][3]));
  }
}

/**
 * Generate random float number in [min, max]
 */
float generateRandomFloat(float min, float max) {
  return min + (rand() / (float)RAND_MAX) * (max - min);
}