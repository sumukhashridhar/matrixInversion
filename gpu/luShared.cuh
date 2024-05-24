#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef double FpType;

__device__ void comp_U(FpType* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        int l = 0;
        FpType sum = 0.0;

        #pragma unroll
        for (l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }
        A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
    }

    else {
        return;
    }
}

__device__ void comp_L(FpType* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum > rowIdx && threadNum < matrixSize) {
        int l = 0;
        FpType sum = 0.0;

        #pragma unroll
        for (l = 0; l < rowIdx; l++) {
            sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }
        A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
    }

    else {
        return;
    }
}

__global__ void lu_decomp(FpType* A, int matrixSize) {

    extern __shared__ FpType shmem[];
    FpType *sh_A = &shmem[0];

    for (int k = threadIdx.x; k < matrixSize * matrixSize; k += blockDim.x) {
        sh_A[k] = A[k];
    }

    __syncthreads();

    for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
        for (int threadNum = threadIdx.x; threadNum < matrixSize; threadNum += blockDim.x) {
            comp_U(sh_A, rowIdx, threadNum, matrixSize);
        }

        __syncthreads();

        for (int threadNum = threadIdx.x; threadNum < matrixSize; threadNum += blockDim.x) {
            comp_L(sh_A, rowIdx, threadNum, matrixSize);
        }

        __syncthreads();
    }

    for (int k = threadIdx.x; k < matrixSize * matrixSize; k += blockDim.x) {
        A[k] = sh_A[k];
    }

    __syncthreads();
}
