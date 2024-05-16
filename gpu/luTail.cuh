#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef double FpType;

__device__ void comp_U(FpType* A, FpType* L, FpType* U, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
    int l=0;
    FpType sum = 0.0;

    #pragma unroll
    for (l = 0; l < rowIdx; l++) {
        sum += L[rowIdx * matrixSize + l] * U[l * matrixSize + threadNum];
    }
    U[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
    }

    else {
        return;
    }
}

__device__ void comp_L(FpType* A, FpType* L, FpType* U, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
    int l=0;
    FpType sum = 0.0;

    if (rowIdx==threadNum) {
        L[threadNum * matrixSize + threadNum] = 1.0;
    }
    else {
        #pragma unroll
        for (l = 0; l < rowIdx; l++) {
            sum += L[threadNum * matrixSize + l] * U[l * matrixSize + rowIdx];
        }
        L[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / U[rowIdx * matrixSize + rowIdx];
    }
    }

    else {
        return;
    }
}

__global__ void lu_decomp(FpType* A, FpType* L, FpType* U, int matrixSize) {
    
    for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
        for (int j = threadIdx.x; j < matrixSize; j += blockDim.x) {
            comp_U(A, L, U, rowIdx, j, matrixSize);
        }

        __syncthreads();

        for (int j = threadIdx.x; j < matrixSize; j += blockDim.x) {
            comp_L(A, L, U, rowIdx, j, matrixSize);
        }

        __syncthreads();
    }
}
