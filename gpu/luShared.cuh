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

    extern __shared__ FpType shmem[];
    FpType *sh_A = &shmem[0];
    FpType *sh_L = &shmem[matrixSize * matrixSize];
    FpType *sh_U = &shmem[2 * matrixSize * matrixSize];

    for (int k = threadIdx.x; k < matrixSize * matrixSize; k += blockDim.x) {
        sh_A[k] = A[k];
        sh_L[k] = L[k];
        sh_U[k] = U[k];
    }

    __syncthreads();

    for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
        for (int threadNum = threadIdx.x; threadNum < matrixSize; threadNum += blockDim.x) {
            comp_U(sh_A, sh_L, sh_U, rowIdx, threadNum, matrixSize);
        }

        __syncthreads();

        for (int threadNum = threadIdx.x; threadNum < matrixSize; threadNum += blockDim.x) {
            comp_L(sh_A, sh_L, sh_U, rowIdx, threadNum, matrixSize);
        }

        __syncthreads();
    }

    for (int k = threadIdx.x; k < matrixSize * matrixSize; k += blockDim.x) {
        A[k] = sh_A[k];
        L[k] = sh_L[k];
        U[k] = sh_U[k];
    }

    __syncthreads();
}
