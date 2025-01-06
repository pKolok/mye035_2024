#include <stdio.h>
#include "DataGenerator.h"
#include "KMeansBuilder.h"

int main() {

   // Create clustering data
  float clusterData[1000][2];
  createClusteringData(clusterData);

  // Write clustering data to files
  exportClusteringData(clusterData);

  // ----- Build k-means model -----
  buildKMeansModel(clusterData);

  printf("=================================================================\n");
	printf("======================== Program finished =======================\n");
  printf("=================================================================\n");
	return 0;
}