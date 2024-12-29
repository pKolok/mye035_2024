#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "ModelBuilderPT3.h"
#include "dataGenerator.h"

#define D 2
#define K 4
#define H1 3
#define H2 3
#define H3 3
#define activation_function "tanh"
// #define activation_function "relu"

static float w_h1[H1][D];
static float w0_h1[H1];
static float u_h1[H1] = { 0 };
static float y_h1[H1] = { 0 };
static float w_h2[H2][H1];
static float w0_h2[H2];
static float u_h2[H2] = { 0 };
static float y_h2[H2] = { 0 };
static float w_h3[H3][H2];
static float w0_h3[H3];
static float u_h3[H3] = { 0 };
static float y_h3[H3] = { 0 };
static float w_exit[K][H3];
static float w0_exit[K];
static float u_exit[K] = { 0 };
static float y_exit[K] = { 0 };

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

  // Initialise level 3 weights (w_h3 & w0_h3)
  for (int i = 0; i < H3; ++i) {
    w0_h3[i] = generateRandomFloat(-1, 1);
    for (int j = 0; j < H2; ++j) {
      w_h3[i][j] = generateRandomFloat(-1, 1);
    }
  }

  // Initialise pt2 & pt3 exit weights (w_exit_pt2 & w_exit)
  for (int i = 0; i < K; ++i) {
    w0_exit[i] = generateRandomFloat(-1, 1);
    for (int k = 0; k < H3; ++k) {
      w_exit[i][k] = generateRandomFloat(-1, 1);
    }
  }

  return;
}

static void forwardPass(float *x, int d, float *y, int k) {
  // Level 1 (tanh or reLu)
  for (int i = 0; i < H1; ++i) {
    for (int j = 0; j < d; ++j) {
      u_h1[i] += x[j] * w_h1[i][j];
    }
    u_h1[i] += w0_h1[i];
    if (strcmp(activation_function, "tanh") == 0) {
      y_h1[i] = tanh(u_h1[i]);
    } else if (strcmp(activation_function, "relu") == 0) {
      y_h1[i] = u_h1[i] > 0 ? u_h1[i] : 0;
    } else {
      exit(-1);
    }
  }

  // Level 2 (tanh or reLu)
  for (int i = 0; i < H2; ++i) {
    for (int j = 0; j < H1; ++j) {
      u_h2[i] += y_h1[j] * w_h2[i][j];
    }
    u_h2[i] += w0_h2[i];
    if (strcmp(activation_function, "tanh") == 0) {
      y_h2[i] = tanh(u_h2[i]);
    } else if (strcmp(activation_function, "relu") == 0) {
      y_h2[i] = u_h2[i] > 0 ? u_h2[i] : 0;
    } else {
      exit(-1);
    }
  }

  // Level 2 (tanh or reLu)
  for (int i = 0; i < H3; ++i) {
    for (int j = 0; j < H2; ++j) {
      u_h3[i] += y_h2[j] * w_h3[i][j];
    }
    u_h3[i] += w0_h3[i];
    if (strcmp(activation_function, "tanh") == 0) {
      y_h3[i] = tanh(u_h3[i]);
    } else if (strcmp(activation_function, "relu") == 0) {
      y_h3[i] = u_h3[i] > 0 ? u_h3[i] : 0;
    } else {
      exit(-1);
    }
  }

  // Output Layer (softmax)
  float u_exit_sum = 0.0;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < H2; ++j) {
      u_exit[i] += y_h3[j] * w_exit[i][j];
    }
    u_exit[i] += w0_exit[i];
    u_exit_sum += exp(u_exit[i]);
  }
  for (int i = 0; i < k; ++i) {
    y[i] = exp(u_exit[i]) / u_exit_sum;
  }
}

void build3LayerNetwork(float data[4000][3]) {
  initialiseWeights();

  forwardPass(data[0], 2, y_exit, 4);
}