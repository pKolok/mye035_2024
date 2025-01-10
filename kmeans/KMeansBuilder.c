#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "KMeansBuilder.h"

#define M 12
#define N 1000

struct Point {
  float x;
  float y;
};

static struct Point centers[M];
static struct Point groups[M][N];
static int groupSizes[M] = { 0 };

void initialiseCenters(float clusterData[N][2]) {
  // Ensure different random generations at each program run
  // srand(time(NULL));

  int generatedIndeces[1000] = { 0 };
  int count = 0;

  while (count < M) {
    int randomPointIndex = rand() % N;

    // Need to make sure that no duplicate indeces are generated
    if (generatedIndeces[randomPointIndex] == 0) {
      generatedIndeces[randomPointIndex] = 1;
      centers[count].x = clusterData[randomPointIndex][0];
      centers[count].y = clusterData[randomPointIndex][1];
      count++;
    }
  }
}

float calculateDistance(struct Point p1, struct Point p2) {
  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  return sqrt(dx * dx + dy * dy);
}

void calculateCenterDistancesFromPoint(struct Point point, float distances[M],
int* nearestCenterIndex) {
  for (int j = 0; j < M; ++j) {
    float distance = calculateDistance(point, centers[j]);
    distances[j] = distance;
    
    if (distance < distances[*nearestCenterIndex]) {
      *nearestCenterIndex = j;
    }
  }
}

int arePointsDifferent(struct Point p1, struct Point p2) {
  if (p1.x != p2.x || p1.y != p2.y) {
    return 1;
  }
  return 0;
}

int updateCenters() {
  int haveCentersChanged = 0;
  
  for (int group = 0; group < M; ++group) {
    int groupSize = groupSizes[group];
    
    float xSum = 0.0;
    float ySum = 0.0;
    for (int p = 0; p < groupSize; ++p) {
      xSum += groups[group][p].x;
      ySum += groups[group][p].y;
    }

    struct Point newGroupCenter = { xSum/groupSize, ySum/groupSize };

    // Change flag if at least one group's centers have changed
    if (arePointsDifferent(centers[group], newGroupCenter)) {
      haveCentersChanged = 1;
    }

    centers[group] = newGroupCenter;
  }

  return haveCentersChanged;
}

float calculateClusteringError() {
  float error = 0.0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < groupSizes[i]; ++j) {
      error += calculateDistance(groups[i][j], centers[i]);
    }
  }
  return error;
}

void exportCenterCoordinates(float centerCoords[M][2]) {
  char *filename = (char *)malloc(22 * sizeof(char));
  if (filename == NULL) {
    printf("Memory allocation failed!\n");
    return;
  }

  snprintf(filename, 22, "ClusterCenters_%d.txt", M);

  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    printf("Error opening file\n");
    return;
  }

  for (int c = 0; c < M; ++c) {
    fprintf(file, "%f %f\n", centerCoords[c][0], centerCoords[c][1]);
  }

  free(filename);
}

void buildKMeansModel(float clusterData[N][2]) {
  printf("-----------------------------------------------------------------\n");
  printf("----------------- Executing k-means for M = %d ------------------\n",
  M);
  printf("-----------------------------------------------------------------\n");

  float minError = FLT_MAX;
  int minErrorRun = 0;
  float centerCoords[M][2] = { 0 };
  
  // Program will be run 20 times
  for (int run = 0; run < 20; ++run) {
    initialiseCenters(clusterData);

    int haveCentersChanged = 0;
    int numberOfIterations = 0;

    do {
      memset(groupSizes, 0, sizeof(groupSizes));
      numberOfIterations++;

      for (int i = 0; i < N; ++i) {
        struct Point point = { clusterData[i][0], clusterData[i][1] };

        float distances[M] = { 0 };
        int nearestCenterIndex = 0;
        calculateCenterDistancesFromPoint(point, distances, &nearestCenterIndex);

        // Assign point to group
        groups[nearestCenterIndex][groupSizes[nearestCenterIndex]] = point;
        groupSizes[nearestCenterIndex]++;
      }

      haveCentersChanged = updateCenters();
    } while (haveCentersChanged == 1);
    
    float error = calculateClusteringError();
    printf("Run %d. Iterations: %d Clustering Error: %f\n", run,
      numberOfIterations, error);

    if (error < minError) {
      minError = error;
      minErrorRun = run;
      for (int c = 0; c < M; ++c) {
        centerCoords[c][0] = centers[c].x;
        centerCoords[c][1] = centers[c].y;
      }
    }
  }

  printf("--------------\n");
  printf(">>>>> The minimum clustering error is %f for run %d <<<<<\n",
    minError, minErrorRun);
  exportCenterCoordinates(centerCoords);

}