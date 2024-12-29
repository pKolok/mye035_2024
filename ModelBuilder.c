#include "ModelBuilder.h"

void transformData(float inputData[4000][3], float outputData[4000][7]) {
  for (int i = 0; i < 4000; ++i) {
    outputData[i][0] = inputData[i][0];
    outputData[i][1] = inputData[i][1];
    outputData[i][2] = inputData[i][2];

    if (inputData[i][2] == 1) {
      outputData[i][3] = 1;
    } else if (inputData[i][2] == 2) {
      outputData[i][4] = 1;
    } else if (inputData[i][2] == 3) {
      outputData[i][5] = 1;
    } else if (inputData[i][2] == 4) {
      outputData[i][6] = 1;
    }
  }
}