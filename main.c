#include <stdio.h>
#include "dataGenerator.h"
#include "ModelBuilderPT2.h"
#include "ModelBuilderPT3.h"
#include "KMeansBuilder.h"

int main() {

  // Create train & test  classification data
  float trainData[4000][3];
	createClassificationData(trainData);
	float testData[4000][3];
	createClassificationData(testData);

  // Write classification data to files
  exportClassificationData(trainData, "ClassificationTrainData.txt");
  exportClassificationData(testData, "ClassificationTestData.txt");

  // Create clustering data
  float clusterData[1000][2];
  createClusteringData(clusterData);

  // Write clustering data to files
  exportClusteringData(clusterData);


  // ----- Build classification model(s) -----
  // build2LayerNetwork(trainData, testData);
  // build3LayerNetwork(trainData, testData);


  // ----- Build k-means model -----
  buildKMeansModel(clusterData);

  printf("=================================================================\n");
	printf("======================== Program finished =======================\n");
  printf("=================================================================\n");
	return 0;
}