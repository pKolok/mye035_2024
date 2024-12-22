#include <stdlib.h>
#include <stdio.h>
#include "dataGenerator.h"

void createClassificationData(float data[4000][3]) {
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

void createClusteringData(float data[1000][2]) {
	for (int i = 0; i < 100; ++i) {
		data[i][0] = generateRandomFloat(-2.0, -1.6);
		data[i][1] = generateRandomFloat(1.6, 2.0);
	};
  for (int i = 100; i < 200; ++i) {
		data[i][0] = generateRandomFloat(-1.2, -0.8);
		data[i][1] = generateRandomFloat(1.6, 2.0);
	};
  for (int i = 200; i < 300; ++i) {
		data[i][0] = generateRandomFloat(-0.4, 0.0);
		data[i][1] = generateRandomFloat(1.6, 2.0);
	};
  for (int i = 300; i < 400; ++i) {
		data[i][0] = generateRandomFloat(-1.8, -1.4);
		data[i][1] = generateRandomFloat(0.8, 1.2);
	};
  for (int i = 400; i < 500; ++i) {
		data[i][0] = generateRandomFloat(-0.6, -0.2);
		data[i][1] = generateRandomFloat(0.8, 1.2);
	};
  for (int i = 500; i < 600; ++i) {
		data[i][0] = generateRandomFloat(-2.0, -1.6);
		data[i][1] = generateRandomFloat(0, 0.4);
	};
  for (int i = 600; i < 700; ++i) {
		data[i][0] = generateRandomFloat(-1.2, -0.8);
		data[i][1] = generateRandomFloat(0, 0.4);
	};
  for (int i = 700; i < 800; ++i) {
		data[i][0] = generateRandomFloat(-0.4, 0);
		data[i][1] = generateRandomFloat(0, 0.4);
	};
  for (int i = 800; i < 1000; ++i) {
		data[i][0] = generateRandomFloat(-2.0, 0);
		data[i][1] = generateRandomFloat(0, 2.0);
	};
}

void exportClassificationData(float data[4000][3]) {
  FILE *file = fopen("ClassificationData.txt", "w");
  if (file == NULL) {
    printf("Error opening file\n");
    return;
  }

  for (int i = 0; i < 4000; ++i) {
    fprintf(file, "%f %f %d\n", data[i][0], data[i][1], (int)(data[i][2]));
  }
}

void exportClusteringData(float data[1000][2]) {
  FILE *file = fopen("ClusteringData.txt", "w");
  if (file == NULL) {
    printf("Error opening file\n");
    return;
  }

  for (int i = 0; i < 1000; ++i) {
    fprintf(file, "%f %f\n", data[i][0], data[i][1]);
  }
}

/**
 * Generate random float number in [min, max]
 */
float generateRandomFloat(float min, float max) {
  return min + (rand() / (float)RAND_MAX) * (max - min);
}

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