#include <stdio.h>
#include "DataGenerator.h"
#include "ModelBuilderPT2.h"
#include "ModelBuilderPT3.h"

int main() {

  // Create train & test  classification data
  float trainData[4000][3];
	createClassificationTrainData(trainData);
	float testData[4000][4];
	createClassificationTestData(testData);

  // Write classification data to files
  exportClassificationTrainData(trainData, "ClassificationTrainData.txt");
  exportClassificationTestData(testData, "ClassificationTestData.txt");

   // ----- Build classification model(s) -----
  build2LayerNetwork(trainData, testData);
  build3LayerNetwork(trainData, testData);

  exportTestData(testData, "ClassificationTestResults.txt");

  printf("=================================================================\n");
	printf("======================== Program finished =======================\n");
  printf("=================================================================\n");
	return 0;
}