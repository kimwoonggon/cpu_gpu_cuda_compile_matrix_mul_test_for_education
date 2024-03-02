#include <cuda_runtime.h>
__global__ void matrixMultiplyKernel(float *a, float *b, float *c, int N);
void gpuMatrixMultiply(float *a, float *b, float *c, int N);