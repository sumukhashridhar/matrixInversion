#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.h"

__device__ void comp_U(FpType* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        FpType sum = 0.0;

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
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
        FpType sum = 0.0;

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }
        A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
    }

    else {
        return;
    }
}

__device__ void inversion(FpType* A, int rowIdx, int matrixSize) {
    FpType elem = 0.0;
    FpType e[32];
    FpType y[32];
    // init e on registers
    #pragma unroll
    for (int i = 0; i < matrixSize; i++) {
        e[i] = (i == rowIdx) ? 1.0 : 0.0;
        y[i] = 0.0;
    }

    // forward substitution
    for (int i = 0; i < matrixSize; i++) {
        FpType sum = 0.0;
        #pragma unroll
        for (int j = 0; j < i; j++) {
            elem = (i == j) ? 1.0 : A[(i * matrixSize) + j];
            sum += elem * y[j];
        }
        y[i] = (e[i] - sum) / 1.0;
    }

    // backward substitution
    for (int i = matrixSize - 1; i >= 0; i--) {
        FpType sum = 0.0;
        #pragma unroll
        for (int j = i + 1; j < matrixSize; j++) {
            sum += A[(i * matrixSize) + j] * A[j * matrixSize + rowIdx];
        }
        A[i * matrixSize + rowIdx] = (y[i] - sum) / A[(i * matrixSize) + i]; 
    }

}

__global__ void batched_lu(FpType* A, int matrixSize, int numMatrices) {
    int blockNum = blockIdx.x;

    if (blockNum >= 0 && blockNum < numMatrices) { 
        int numElements = matrixSize * matrixSize;
        int offset = blockNum * numElements;

        extern __shared__ FpType shmem[];
        FpType *sh_A = &shmem[0];

        for (int k = threadIdx.x; k < numElements; k += blockDim.x) {
            sh_A[k] = A[k + offset];
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

        for (int rowIdx = threadIdx.x; rowIdx < matrixSize; rowIdx += blockDim.x) {
            if (rowIdx < matrixSize) {
                inversion(sh_A, rowIdx, matrixSize);
            }   
        }

        __syncthreads();

        for (int k = threadIdx.x; k < numElements; k += blockDim.x) {
            A[k + offset] = sh_A[k];
        }

        __syncthreads();

    }

    __syncthreads();

}
