#include <stdio.h>
#include "dataGenerator.h"
#include "ModelBuilder.h"

int main() {

  // Create train & test  classification data
  float trainData[4000][3];
	createClassificationData(trainData);
	float testData[4000][3];
	createClassificationData(testData);

  // Create clustering data
  float clusterData[1000][2];
  createClusteringData(clusterData);

  // Write data to files
  exportClassificationData(trainData);
  exportClusteringData(clusterData);


  // ----- Build model(s) -----
  build2LayerNetwork(trainData);


	printf("Program finished");
	return 0;
}