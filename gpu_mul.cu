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


__global__ void matrixMultiplySharedKernel(float *a, float *b, float *c, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row and column of the d_C element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int ph = 0; ph < ceil(N/(float)TILE_WIDTH); ++ph) {
        // Load the A and B tiles into shared memory
        if (Row < N && ph*TILE_WIDTH+tx < N)
            sA[ty][tx] = a[Row*N + ph*TILE_WIDTH+tx];
        else
            sA[ty][tx] = 0.0;

        if (Col < N && ph*TILE_WIDTH+ty < N)
            sB[ty][tx] = b[(ph*TILE_WIDTH+ty)*N + Col];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (Row < N && Col < N)
        c[Row*N + Col] = sum;
}

void gpuMatrixMultiplyShared(float *a, float *b, float *c, int N) {
    float *dev_a, *dev_b, *dev_c;
    
    cudaMalloc((void**)&dev_a, N*N*sizeof(float));
    cudaMalloc((void**)&dev_b, N*N*sizeof(float));
    cudaMalloc((void**)&dev_c, N*N*sizeof(float));
    
    cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMultiplySharedKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    
    cudaMemcpy(c, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
