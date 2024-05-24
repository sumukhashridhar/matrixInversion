#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef double FpType;

__device__ void comp_U(FpType* A, FpType* L, FpType* U, int rowIdx, int threadNum, int offset, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        FpType sum = 0.0;

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += L[(rowIdx * matrixSize) + offset + l] * U[(l * matrixSize) + offset + threadNum];
        }
        U[(rowIdx * matrixSize) + offset + threadNum] = A[(rowIdx * matrixSize) + offset + threadNum] - sum;
    }

    else {
        return;
    }
}

__device__ void comp_L(FpType* A, FpType* L, FpType* U, int rowIdx, int threadNum, int offset, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        FpType sum = 0.0;

        if (rowIdx==threadNum) {
            L[(threadNum * matrixSize) + offset + threadNum] = 1.0;
        }
        
        else {
            #pragma unroll
            for (int l = 0; l < rowIdx; l++) {
                sum += L[(threadNum * matrixSize) + offset + l] * U[(l * matrixSize) + offset + rowIdx];
            }
            L[(threadNum * matrixSize) + offset + rowIdx] = (A[(threadNum * matrixSize) + offset + rowIdx] - sum) / U[(rowIdx * matrixSize) + offset + rowIdx];
        }
    }

    else {
        return;
    }
}

__global__ void batched_lu(FpType* A, FpType* L, FpType* U, int matrixSize, int numMatrices) {
    int blockNum = blockIdx.x;

    if (blockNum >= 0 && blockNum < numMatrices) { 
        int numElements = matrixSize * matrixSize;
        int offset = blockNum * numElements;

        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int threadNum = threadIdx.x; threadNum < matrixSize; threadNum += blockDim.x) {
                comp_U(A, L, U, rowIdx, threadNum, offset, matrixSize);
            }

            __syncthreads();

            for (int threadNum = threadIdx.x; threadNum < matrixSize; threadNum += blockDim.x) {
                comp_L(A, L, U, rowIdx, threadNum, offset, matrixSize);
            }

            __syncthreads();
        }

    }

    __syncthreads();

}
