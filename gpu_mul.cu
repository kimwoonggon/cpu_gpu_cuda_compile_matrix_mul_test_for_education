#include "gpu_mul.h"

__global__ void matrixMultiplyKernel(float *a, float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

void gpuMatrixMultiply(float *a, float *b, float *c, int N) {
    float *dev_a, *dev_b, *dev_c;
    
    cudaMalloc((void**)&dev_a, N*N*sizeof(float));
    cudaMalloc((void**)&dev_b, N*N*sizeof(float));
    cudaMalloc((void**)&dev_c, N*N*sizeof(float));
    
    cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    
    cudaMemcpy(c, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
