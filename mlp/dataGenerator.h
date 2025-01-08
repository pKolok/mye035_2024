#ifndef DATA_GENERATOR
#define DATA_GENERATOR

void createClassificationTrainData(float[4000][3]);
void createClassificationTestData(float[4000][4]);
void exportClassificationTrainData(float[4000][3], char* );
void exportClassificationTestData(float[4000][4], char* );
void exportTestData(float[4000][4], char* );

float generateRandomFloat(float, float);

#endif