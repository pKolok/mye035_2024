#include <stdlib.h>
#include <stdio.h>
#include "DataGenerator.h"

/**
 * Generate random float number in [min, max]
 */
float generateRandomFloat(float min, float max) {
  return min + (rand() / (float)RAND_MAX) * (max - min);
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
