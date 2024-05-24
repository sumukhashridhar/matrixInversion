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

__device__ void forward_sub(FpType* L, FpType* e, FpType* y, int matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        FpType sum = 0.0;
        #pragma unroll
        for (int j = 0; j < i; j++) {
            sum += L[i * matrixSize + j] * y[j];
        }
        y[i] = (e[i] - sum) / L[i * matrixSize + i];
    }
}

__device__ void backward_sub(FpType* U, FpType* y, FpType* A_inv, int matrixSize) {
    for (int i = matrixSize - 1; i >= 0; i--) {
        FpType sum = 0.0;
        #pragma unroll
        for (int j = i + 1; j < matrixSize; j++) {
            sum += U[i * matrixSize + j] * A_inv[j];
        }
        A_inv[i] = (y[i] - sum) / U[i * matrixSize + i];
    }
}

__global__ void lu_decomp(FpType* A, FpType* L, FpType* U, FpType* e, FpType* y, FpType* A_inv, int matrixSize) {
    
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

    for (int rowIdx = threadIdx.x; rowIdx < matrixSize; rowIdx =+ blockDim.x) {
        // e[rowIdx * matrixSize + rowIdx] = 1.0;
        forward_sub(L, &e[rowIdx * matrixSize], &y[rowIdx * matrixSize], matrixSize);
        backward_sub(U, &y[rowIdx * matrixSize], &A_inv[rowIdx * matrixSize], matrixSize);
    }

    __syncthreads();
}
