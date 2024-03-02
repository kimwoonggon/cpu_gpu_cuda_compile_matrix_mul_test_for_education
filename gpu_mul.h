#include <cuda_runtime.h>
#define TILE_WIDTH 32
__global__ void matrixMultiplyKernel(float *a, float *b, float *c, int N);
void gpuMatrixMultiply(float *a, float *b, float *c, int N);
__global__ void matrixMultiplySharedKernel(float *a, float *b, float *c, int N);
void gpuMatrixMultiplyShared(float *a, float *b, float *c, int N);

