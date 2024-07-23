#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"

__device__ void comp_U(FpType* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        FpType sum = 0.0;

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }
        A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
        // printf("A[%d][%d]: %f\n", rowIdx, threadNum, A[rowIdx * matrixSize + threadNum]);
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
        // printf("A[%d][%d]: %f\n", threadNum, rowIdx, A[threadNum * matrixSize + rowIdx]);
    }

    else {
        return;
    }
}

/*
__device__ void forward_substitution(FpType* A, FpType* y, int rowIdx, int matrixSize) {
    if (rowIdx < matrixSize) {
        // printf("rowIdx: %d\n", rowIdx);
        
        FpType elem = 0.0;
        // FpType y[128];
        // init y on registers
        // #pragma unroll
        // for (int i = 0; i < matrixSize; i++) {
        //     y[i] = 0.0;
        // }

        // forward substitution
        for (int i = 0; i < matrixSize; i++) {
            FpType sum = 0.0;
            #pragma unroll
            for (int j = 0; j < i; j++) {
                elem = (i == j) ? 1.0 : A[(i * matrixSize) + j];
                sum += elem * y[j];
            }
            y[i] = (((i == rowIdx) ? 1.0 : 0.0) - sum);
        }
    }
    
    else {
        return;
    }
}

__device__ void backward_substitution(FpType* A, FpType* y, int rowIdx, int matrixSize) {
    if (rowIdx < matrixSize) {
        // for (int i = 0; i < matrixSize; i++) {
        //     printf("y[%d]: %f\n", i, y[i]);
        // }

        // backward substitution
        for (int i = matrixSize - 1; i >= 0; i--) {
            FpType sum = 0.0;
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + rowIdx];
            }
            A[i * matrixSize + rowIdx] = (y[i] - sum) / A[(i * matrixSize) + i]; 
        }
    }

    else {
        return;
    }

}
*/

__device__ void inversion(FpType* A, int rowIdx, int matrixSize) {
    if (rowIdx < matrixSize) {
                // printf("rowIdx: %d\n", rowIdx);

        FpType y[128];
   
        // forward substitution
        for (int i = 0; i < matrixSize; i++) { // take care of the indices like rowIdx
            FpType sum = 0.0;
            #pragma unroll
            for (int j = 0; j < i; j++) {
                // elem = (i == j) ? 1.0 : A[(i * matrixSize) + j];
                // sum += elem * y[j];
                sum += A[(i * matrixSize) + j] * y[j];
            }
            y[i] = (((i == rowIdx) ? 1.0 : 0.0) - sum);
        }

        // backward substitution
        for (int i = matrixSize - 1; i >= 0; i--) {
            FpType sum = 0.0;
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + rowIdx];
            }
            A[(i * matrixSize) + rowIdx] = (y[i] - sum) / A[(i * matrixSize) + i]; 
        }
    }

    else {
        return;
    }
}

__device__ int calc_swl_idx(int k, int matrixSize, int numElements) {
    // impl has div and mod, exps on GPU
    int count = k / matrixSize;
    printf("count is: %d\n", count);
    return (k * matrixSize) % numElements + (k / matrixSize);
}


__global__ void batched_lu(FpType* A, int matrixSize, int numMatrices) {
    int blockNum = blockIdx.x;

    if (blockNum >= 0 && blockNum < numMatrices) { 
        int numElements = matrixSize * matrixSize;
        int offset = blockNum * numElements;

        extern __shared__ FpType shmem[];
        FpType *sh_A = &shmem[0];
        // int count = 0;

        for (int k = threadIdx.x; k < numElements; k += blockDim.x) {
            int swl_idx = calc_swl_idx(k, matrixSize, numElements);
            sh_A[swl_idx] = A[swl_idx + offset];
            // count++;
            // printf("sh_A[%d]: %f\n", swl_idx, sh_A[swl_idx]);
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
            // FpType y[128];
            // forward_substitution(sh_A, y, rowIdx, matrixSize);
            // __syncthreads();
            // backward_substitution(sh_A, y, rowIdx, matrixSize);
            // int col = rowIdx + threadIdx.x;
            if (rowIdx < matrixSize) {
                // printf("blockIdx: %d rowIdx: %d\n", blockNum, rowIdx);

                inversion(sh_A, rowIdx, matrixSize);
            }
            // __syncthreads();
        }

        __syncthreads();

        // count = 0;
        for (int k = threadIdx.x; k < numElements; k += blockDim.x) {
            int swl_idx = calc_swl_idx(k, matrixSize, numElements);
            A[swl_idx + offset] = sh_A[swl_idx];
            // count++;
            // printf("A[%d]: %f\n", swl_idx + offset, A[swl_idx + offset]);
        
        }

        __syncthreads();

    }

    __syncthreads();

}
