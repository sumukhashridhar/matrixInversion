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

__global__ void batched_lu(FpType* A, FpType* L, FpType* U, int matrixSize, int numMatrices) {
    int blockNum = blockIdx.x;

    if (blockNum >= 0 && blockNum < numMatrices) { 
        int numElements = matrixSize * matrixSize;
        int offset = blockNum * numElements;

        extern __shared__ FpType shmem[];
        FpType *sh_A = &shmem[0];
        FpType *sh_L = &shmem[numElements];
        FpType *sh_U = &shmem[2 * numElements];

        for (int k = threadIdx.x; k < numElements; k += blockDim.x) {
            sh_A[k] = A[offset + k];
            sh_L[k] = L[offset + k];
            sh_U[k] = U[offset + k];
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

        for (int k = threadIdx.x; k < numElements; k += blockDim.x) {
            A[offset + k] = sh_A[k];
            L[offset + k] = sh_L[k];
            U[offset + k] = sh_U[k];
        }

        __syncthreads();
    }
}

/*
__global__ void batched_lu(FpType* Ain, FpType* Lin, FpType* Uin, int n, int numMatrices) {
    int blockNum = blockIdx.x;

    if (blockNum >= 0 && blockNum < numMatrices) {
        extern __shared__ FpType shmem[];
        FpType *A = &shmem[0];
        FpType *L = &shmem[n * n];
        FpType *U = &shmem[2 * n * n];

        int matrixSize = n;
        int blockSize = blockDim.x;
        int threadNum = threadIdx.x;
        int ratio = matrixSize / blockSize;
        int rem = matrixSize % blockSize;
        int m = ratio - 1;
        int shift_rem = threadNum + blockSize * m + rem;
        int offsetIdx = (blockNum * numMatrices * matrixSize);
        // printf("Offset idx: %d\n", offsetIdx);

        // copy A, L, U to shared memory
        if (threadNum < matrixSize) {
            for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
                A[rowIdx * matrixSize + offset + threadNum] = Ain[rowIdx * matrixSize + offset + threadNum];
                L[rowIdx * matrixSize + offset + threadNum] = Lin[rowIdx * matrixSize + offset + threadNum];
                U[rowIdx * matrixSize + offset + threadNum] = Uin[rowIdx * matrixSize + offset + threadNum];
            }
        }

        __syncthreads();

        if (rem > 0 && rem < blockSize) {
            for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
                if (shift_rem >= rowIdx && shift_rem < matrixSize){
                    A[rowIdx * matrixSize + offset + shift_rem] = Ain[rowIdx * matrixSize + offset + shift_rem];
                    L[rowIdx * matrixSize + offset + shift_rem] = Lin[rowIdx * matrixSize + offset + shift_rem];
                    U[rowIdx * matrixSize + offset + shift_rem] = Uin[rowIdx * matrixSize + offset + shift_rem];
                }
            }
        }

        __syncthreads();

        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            if (threadNum < matrixSize) {
                // printf("Block num: %d, Thread num: %d, Row idx: %d\n", blockNum, threadNum, rowIdx);
                int k = 0;
                for (k = 0; k < ratio; k++) {
                    comp_U(A, L, U, rowIdx, threadNum + blockSize * k, matrixSize);
                    // comp_U(A, L, U, rowIdx, offsetIdx, threadNum + blockSize * k, matrixSize);
                    // sum += L[(rowIdx * matrixSize) + (blockNum * numMatrices * matrixSize) + k] * U[(k * matrixSize) + (blockNum * numMatrices * matrixSize) + threadNum];
                }
                // U[(rowIdx * matrixSize) + (blockNum * numMatrices * matrixSize) + threadNum] = A[(rowIdx * matrixSize) + (blockNum * numMatrices * matrixSize) + threadNum] - sum;
            }

            if (rem > 0 && rem < blockSize) {
                if (shift_rem >= rowIdx && shift_rem < matrixSize){
                    comp_U(A, L, U, rowIdx, shift_rem, matrixSize);
                }
            }

            __syncthreads();

            if (threadNum < matrixSize) {
                int k = 0;
                for (k = 0; k <= ratio; k++) {
                    comp_L(A, L, U, rowIdx, threadNum + blockSize * k, matrixSize);
                    // comp_L(A, L, U, rowIdx, offsetIdx, threadNum + blockSize * k, matrixSize);
                }
                // if (rowIdx==threadNum) {
                //     L[(threadNum * matrixSize) + (blockNum * numMatrices * matrixSize) + threadNum] = 1.0;
                // }
                // else {
                //     FpType sum = 0.0;
                //     for (k = 0; k < i; k++) {
                //         sum += L[(j * matrixSize) + (b * numMatrices * matrixSize) + k] * U[(k * matrixSize) + (b * numMatrices * matrixSize) + i];
                //     }
                //     L[(j * matrixSize) + (b * numMatrices * matrixSize) + i] = (A[(j * matrixSize) + (b * numMatrices * matrixSize) + i] - sum) / U[(i * matrixSize) + (b * numMatrices * matrixSize) + i];
                // }
            }

            if (rem > 0 && rem < blockSize) {
                if (shift_rem >= rowIdx && shift_rem < matrixSize){
                    comp_L(A, L, U, rowIdx, shift_rem, matrixSize);
                    // comp_L(A, L, U, rowIdx, offsetIdx, shift_rem, matrixSize);
                }
            }

            __syncthreads();
        }

        // copy A, L, U back to global memory
        if (threadNum < matrixSize) {
            for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
                Ain[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum];
                Lin[rowIdx * matrixSize + threadNum] = L[rowIdx * matrixSize + threadNum];
                Uin[rowIdx * matrixSize + threadNum] = U[rowIdx * matrixSize + threadNum];
            }
        }

        __syncthreads();

        if (rem > 0 && rem < blockSize) {
            for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
                if (shift_rem >= rowIdx && shift_rem < matrixSize){
                    Ain[rowIdx * matrixSize + shift_rem] = A[rowIdx * matrixSize + shift_rem];
                    Lin[rowIdx * matrixSize + shift_rem] = L[rowIdx * matrixSize + shift_rem];
                    Uin[rowIdx * matrixSize + shift_rem] = U[rowIdx * matrixSize + shift_rem];
                }
            }
        }
    }
}
*/