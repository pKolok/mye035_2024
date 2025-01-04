#ifndef DATA_GENERATOR
#define DATA_GENERATOR

void createClassificationData(float[4000][3]);
void createClusteringData(float data[1000][2]);
void exportClassificationData(float[4000][3], char* );
void exportClusteringData(float data[1000][2]);

float generateRandomFloat(float, float);
int getClassificationCategory(float, float);

#endif