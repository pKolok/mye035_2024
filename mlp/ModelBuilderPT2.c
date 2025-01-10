#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "ModelBuilder.h"
#include "ModelBuilderPT2.h"
#include "DataGenerator.h"

#define N 4000
#define B 20
#define D 2
#define K 4
#define H1 15
#define H2 15
#define activation_function_Η1 "tanh"
// #define activation_function_Η2 "tanh"
#define activation_function_Η2 "relu"
#define H 0.001
#define THRESSHOLD 0.1

static float train_categories[N][K] = { 0 };
static float test_categories[N][K] = { 0 };
// Layer 1:
static float w_h1[H1][D];
static float w0_h1[H1];
static float u_h1[H1] = { 0 };
static float y_h1[H1] = { 0 };
static float delta_h1[H1] = { 0 };
static float theta_Ew_h1[H1][D] = { 0 };
static float theta_Ew0_h1[H1] = { 0 };
static float theta_Ew_sum_h1[H1][D] = { 0 };
static float theta_Ew0_sum_h1[H1] = { 0 };
// Layer 2:
static float w_h2[H2][H1];
static float w0_h2[H2];
static float u_h2[H2] = { 0 };
static float y_h2[H2] = { 0 };
static float delta_h2[H2] = { 0 };
static float theta_Ew_h2[H2][H1] = { 0 };
static float theta_Ew0_h2[H2] = { 0 };
static float theta_Ew_sum_h2[H2][H1] = { 0 };
static float theta_Ew0_sum_h2[H2] = { 0 };
// Out layer:
static float w_out[K][H2];
static float w0_out[K];
static float u_out[K] = { 0 };
static float y_out[K] = { 0 };
static float delta_out[K] = { 0 };
static float theta_Ew_out[K][H2] = { 0 };
static float theta_Ew0_out[K] = { 0 };
static float theta_Ew_sum_out[K][H2] = { 0 };
static float theta_Ew0_sum_out[K] = { 0 };

/* --------------------- Static (i.e. "private") methods -------------------- */
static void initialiseWeights() {
  // Initialise level 1 weights (w_h1 & w0_h1)
  for (int i = 0; i < H1; ++i) {
    w0_h1[i] = generateRandomFloat(-1, 1);
    for (int j = 0; j < D; ++j) {
      w_h1[i][j] = generateRandomFloat(-1, 1);
    }
  }

  // Initialise level 2 weights (w_h2 & w0_h2)
  for (int i = 0; i < H2; ++i) {
    w0_h2[i] = generateRandomFloat(-1, 1);
    for (int j = 0; j < H1; ++j) {
      w_h2[i][j] = generateRandomFloat(-1, 1);
    }
  }

  // Initialise exit weights (w_out)
  for (int i = 0; i < K; ++i) {
    w0_out[i] = generateRandomFloat(-1, 1);
    for (int j = 0; j < H2; ++j) {
      w_out[i][j] = generateRandomFloat(-1, 1);
    }
  }

  return;
}

static void forwardPass(float *x, int d, float *y, int k) {
  // Initialisations
  memset(u_h1, 0, sizeof(u_h1));
  memset(u_h2, 0, sizeof(u_h2));
  memset(u_out, 0, sizeof(u_out));

  // Level 1 (tanh or reLu)
  for (int i = 0; i < H1; ++i) {
    for (int j = 0; j < d; ++j) {
      u_h1[i] += x[j] * w_h1[i][j];
    }
    u_h1[i] += w0_h1[i];
    if (strcmp(activation_function_Η1, "tanh") == 0) {
      y_h1[i] = tanh(u_h1[i]);
    } else if (strcmp(activation_function_Η1, "relu") == 0) {
      y_h1[i] = u_h1[i] > 0 ? u_h1[i] : 0;
    } else {
	    printf("Invalid Activation Function");
      exit(-1);
    }
  }

  // Level 2 (tanh or reLu)
  for (int i = 0; i < H2; ++i) {
    for (int j = 0; j < H1; ++j) {
      u_h2[i] += y_h1[j] * w_h2[i][j];
    }
    u_h2[i] += w0_h2[i];
    if (strcmp(activation_function_Η2, "tanh") == 0) {
      y_h2[i] = tanh(u_h2[i]);
    } else if (strcmp(activation_function_Η2, "relu") == 0) {
      y_h2[i] = u_h2[i] > 0 ? u_h2[i] : 0;
    } else {
	    printf("Invalid Activation Function");
      exit(-1);
    }
  }

  // Output Layer (softmax)
  float u_exit_sum = 0.0;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < H2; ++j) {
      u_out[i] += y_h2[j] * w_out[i][j];
    }
    u_out[i] += w0_out[i];
    u_exit_sum += exp(u_out[i]);
  }
  for (int i = 0; i < k; ++i) {
    y[i] = exp(u_out[i]) / u_exit_sum;
  }

}

static void backprop(float *x, int d, float *t, int k) {
  // Output Layer (softmax)
  for (int i = 0; i < k; ++i) {
    // Error
    delta_out[i] = y_out[i] - t[i];

    // Error/weight partial derivative
    for (int j = 0; j < H2; ++j) {
      theta_Ew_out[i][j] = delta_out[i] * y_h2[j];
    }
    theta_Ew0_out[i] = delta_out[i];
  }

  // Level 2 (tanh or reLu)
  for (int i = 0; i < H2; ++i) {
    // Error
    float deriv = 0.0;
    if (strcmp(activation_function_Η2, "tanh") == 0) {
      deriv = 1 - pow(tanh(y_h2[i]), 2);
    } else if (strcmp(activation_function_Η2, "relu") == 0) {
      deriv = y_h2[i] > 0 ? 1 : 0;
    } else {
	    printf("Invalid Activation Function");
      exit(-1);
    }
    float w_deltas = 0.0;
    for (int j = 0; j < k; j++ ) {
      w_deltas += w_out[j][i] * delta_out[j];
    }
    delta_h2[i] = deriv * w_deltas;

    // Error/weight partial derivative
    for (int j = 0; j < H1; ++j) {
      theta_Ew_h2[i][j] = delta_h2[i] * y_h1[j];
    }
    theta_Ew0_h2[i] = delta_h2[i];
  }

  // Level 1 (tanh or reLu)
  for (int i = 0; i < H1; ++i) {
    // Error
    float deriv = 0.0;
    if (strcmp(activation_function_Η1, "tanh") == 0) {
      deriv = 1 - pow(tanh(y_h1[i]), 2);
    } else if (strcmp(activation_function_Η1, "relu") == 0) {
      deriv = y_h1[i] > 0 ? 1 : 0;
    } else {
	    printf("Invalid Activation Function");
      exit(-1);
    }
    float w_deltas = 0.0;
    for (int j = 0; j < H2; j++ ) {
      w_deltas += w_h2[j][i] * delta_h2[j];
    }
    delta_h1[i] = deriv * w_deltas;
    
    // Error/weight partial derivative
    for (int j = 0; j < d; ++j) {
      theta_Ew_h1[i][j] = delta_h1[i] * x[j];
    }
    theta_Ew0_h1[i] = delta_h1[i];
  }

}

static void initialiseWeightPartialDerivatives() {
  memset(theta_Ew_sum_h1, 0, sizeof(theta_Ew_sum_h1));
  memset(theta_Ew0_sum_h1, 0, sizeof(theta_Ew0_sum_h1));
  memset(theta_Ew_sum_h2, 0, sizeof(theta_Ew_sum_h2));
  memset(theta_Ew0_sum_h2, 0, sizeof(theta_Ew0_sum_h2));
  memset(theta_Ew_sum_out, 0, sizeof(theta_Ew_sum_out));
  memset(theta_Ew0_sum_out, 0, sizeof(theta_Ew0_sum_out));
}

static void accummulateWeightErrorPartialDerivatives() {
  // Level 1
  for (int i = 0; i < H1; ++i) {
    for (int j = 0; j < D; ++j) {
      theta_Ew_sum_h1[i][j] += theta_Ew_h1[i][j];
    }
    theta_Ew0_sum_h1[i] += theta_Ew0_h1[i];
  }

  // Level 2
  for (int i = 0; i < H2; ++i) {
    for (int j = 0; j < H1; ++j) {
      theta_Ew_sum_h2[i][j] += theta_Ew_h2[i][j];
    }
    theta_Ew0_sum_h2[i] += theta_Ew0_h2[i];
  }

  // Level Out
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < H2; ++j) {
      theta_Ew_sum_out[i][j] += theta_Ew_out[i][j];
    }
    theta_Ew0_sum_out[i] += theta_Ew0_out[i];
  }
}

static void updateWeights() {
  // Level 1
  for (int i = 0; i < H1; ++i) {
    for (int j = 0; j < D; ++j) {
      w_h1[i][j] -= H * theta_Ew_sum_h1[i][j];
    }
    w0_h1[i] -= H * theta_Ew0_sum_h1[i];
  }

  // Level 2
  for (int i = 0; i < H2; ++i) {
    for (int j = 0; j < H1; ++j) {
      w_h2[i][j] -= H * theta_Ew_sum_h2[i][j];
    }
    w0_h2[i] -= H * theta_Ew0_sum_h2[i];
  }

  // Level Out
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < H2; ++j) {
      w_out[i][j] -= H * theta_Ew_sum_out[i][j];
    }
    w0_out[i] -= H * theta_Ew0_sum_out[i];
  }
}

static void train(float trainData[N][3]) {
  printf("-----------------------------------------------------------------\n");
  printf("Training of 2 hidden layers neural network starting...\n");
  printf("-----------------------------------------------------------------\n");

  int epoch = 0;
  float previousEpochError = 0.0;
  float currentEpochError = 0.0;
  float errorDelta = 0.0;
  
  do {
    previousEpochError = currentEpochError;
    currentEpochError = 0.0;

    initialiseWeightPartialDerivatives();

    for (int i = 0; i < N; ++i) {
      forwardPass(trainData[i], D, y_out, K);
      backprop(trainData[i], D, train_categories[i], K);
      accummulateWeightErrorPartialDerivatives();
      
      if ((i+1) % B == 0 || i == N-1) {
        updateWeights();
        initialiseWeightPartialDerivatives();
      }

      float exampleError = 0.0;
      for (int j = 0; j < K; ++j) {
        exampleError += pow(delta_out[j], 2);
      }
      currentEpochError += exampleError;
    }

    currentEpochError /= 2;
    errorDelta = abs(previousEpochError - currentEpochError);

    printf("Epoch: %d. Error: %f. Error delta: %f\n", epoch, currentEpochError,
      errorDelta);

    epoch++;
  } while (epoch < 800 || errorDelta > THRESSHOLD);

  printf("-----------------------------------------------------------------\n");
  printf("Training of 2 hidden layers neural network completed!\n");
  printf("-----------------------------------------------------------------\n");
}

static float test(float testData[N][4]) {
  int correctAnswersCounter = 0;
  float y_test_out[K] = { 0 };

  for (int i = 0; i < N; ++i) {
    forwardPass(testData[i], D, y_test_out, K);

    // Chosen category is the one with the highest percentage
    int maxIndex = 0;
    for (int j = 0; j < K; ++j) {
      if (y_test_out[j] > y_test_out[maxIndex]) {
        maxIndex = j;
      }
    }

    if (test_categories[i][maxIndex] == 1) {
      correctAnswersCounter++;
      testData[i][3] = 1.0;
    }
  }

  float correctDecisionsPercentage = correctAnswersCounter/(float)N;
  printf("The percentage of correct decisions is: %f\n\n",
  correctDecisionsPercentage);
}

/* ---------------------------- Public methods ------------------------------ */
void build2LayerNetwork(float trainData[N][3], float testData[N][4]) {
  extractTrainCategories(trainData, train_categories);
  extractTestCategories(testData, test_categories);

  initialiseWeights();

  // Test forward-pass
  // forwardPass(trainData[0], D, y_out, K);

  // Test backprop
  // backprop(trainData[0], D, train_categories[0], K);
  
  train(trainData);
  test(testData);
}