#include <stdio.h>
#include "DataGenerator.h"
#include "ModelBuilderPT2.h"
#include "ModelBuilderPT3.h"

int main() {

  // Create train & test  classification data
  float trainData[4000][3];
	createClassificationData(trainData);
	float testData[4000][3];
	createClassificationData(testData);

  // Write classification data to files
  exportClassificationData(trainData, "ClassificationTrainData.txt");
  exportClassificationData(testData, "ClassificationTestData.txt");

   // ----- Build classification model(s) -----
  build2LayerNetwork(trainData, testData);
  build3LayerNetwork(trainData, testData);

  printf("=================================================================\n");
	printf("======================== Program finished =======================\n");
  printf("=================================================================\n");
	return 0;
}