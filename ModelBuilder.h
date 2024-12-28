#ifndef MODEL_BUILDER
#define MODEL_BUILDER

void build2LayerNetwork(float[4000][3]);

void initialiseWeights();
void forwardPassPt2(float*, int, float*, int);

#endif