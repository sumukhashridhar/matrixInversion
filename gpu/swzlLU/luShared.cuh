#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef double FpType;

__device__ int calc_swl_idx_lei(int k, int matrixSize) {

    int row = k / matrixSize;
    int col = k % matrixSize;
    int x_swz = (col ^ row);// + (matrixIdInBlock * numElements);

    if (int(k / matrixSize) > 0) {
        x_swz += matrixSize * (k / matrixSize);
    }

    // if (threadIdx.x == 0) {
    //     printf("oldIdx: %d newIdx: %d\n", k, x_swz);
    // }

    return x_swz;
}

__device__ void comp_U(FpType* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        int l = 0;
        FpType sum = 0.0;

        #pragma unroll
        for (l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + threadNum, matrixSize)];
            // sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }

        A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize)] = A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize)] - sum;
        // A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
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
            sum += A[calc_swl_idx_lei(threadNum * matrixSize + l, matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize)];
            // sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize)] = (A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize)] - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize)];
        // A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
    }

    else {
        return;
    }
}

__global__ void lu_decomp(FpType* A, int matrixSize) {

    extern __shared__ FpType shmem[];
    FpType *sh_A = &shmem[0];

    for (int k = threadIdx.x; k < matrixSize * matrixSize; k += blockDim.x) {
        // printf("oldIdx: %d newIdx: %d\n", k, calc_swl_idx_lei(k, matrixSize));
        sh_A[calc_swl_idx_lei(k, matrixSize)] = A[k];
        // sh_A[k] = A[k];
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
        A[k] = sh_A[calc_swl_idx_lei(k, matrixSize)];
        // A[k] = sh_A[k];
    }

    __syncthreads();
}
