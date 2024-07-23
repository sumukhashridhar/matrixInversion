#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

template<typename T>
__device__ void comp_U(T* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        T sum = static_cast<T>(0.0);

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


template<typename T>
__device__ void comp_L(T* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum > rowIdx && threadNum < matrixSize) {
        T sum = static_cast<T>(0.0);

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


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize) {
    if (rowIdx < matrixSize) {
        T y[128];
        // init e on registers
        // #pragma unroll
        // for (int i = 0; i < matrixSize; i++) {
        //     y[i] = 0.0;
        // }

        // forward substitution
        for (int i = 0; i < matrixSize; i++) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = 0; j < i; j++) {
                sum += A[(i * matrixSize) + j] * y[j];
            }
            y[i] = (((i == rowIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        for (int i = matrixSize - 1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
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

// __device__ int calc_swz_idx(int x, int NX) {
//     int sizeof_T = sizeof(float);
//     int sizeof_TC = sizeof_T * 8;
//     int SWIZZLE_SIZE = NX * sizeof_T;

//     int i_chunk = (x * NX * sizeof_T) / sizeof_TC;
//     int y_chunk = i_chunk / (SWIZZLE_SIZE / sizeof_TC);
//     int x_chunk = i_chunk % (SWIZZLE_SIZE / sizeof_TC);
//     // printf("y_chunk: %d\n", y_chunk);
//     // printf("x_chunk: %d\n", x_chunk);

//     int x_chunk_swz = y_chunk ^ x_chunk;
//     // printf("x_chunk_swz: %d\n", x_chunk_swz);

//     int x_swz = (x_chunk_swz * sizeof_TC / sizeof_T) % NX + x % (sizeof_TC / sizeof_T);

//     return x_swz;
// }

__device__ int calc_swl_idx(int k, int matrixSize, int numElements) {
    // impl has div and mod, exps on GPU
    // int count = k / matrixSize;
    // printf("count is: %d\n", count);
    return (k * matrixSize) % numElements + (k / matrixSize);
}

template<typename T>
__global__ void batched_lu_subwarp(T* A, int matrixSize, int numMatrices, int threadsPerMatrix, int matricesPerBlock) {
    int blockNum = blockIdx.x;
    int threadIdInBlock = threadIdx.x;
    int matrixIdInBlock = threadIdInBlock / threadsPerMatrix;
    int globalMatrixId = blockNum * matricesPerBlock + matrixIdInBlock;

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix = threadIdInBlock % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[matrixIdInBlock * numElements];

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            // int row = k / matrixSize;
            // int col = k % matrixSize;   
            // int swz_idx = row * matrixSize + ((col ^ (row & 0x1f)) % matrixSize);
            // // print k row col swz_idx
            // printf("row: %d, col: %d, k: %d, swz_idx: %d\n", row, col, k, swz_idx);
            // int bankOffset = k % matrixSize;
            int swl_idx = calc_swl_idx(k, matrixSize, numElements);

            // int swz_idx = calc_swz_idx(k, numElements * matricesPerBlock);
            // printf("swz_idx: %d, k: %d\n", swz_idx, k);
            sh_A[swl_idx] = A[swl_idx + mtrxOffset];
            // printf("sh_A[%d]: %f\n", k + bankOffset, sh_A[k + bankOffset]);
        }
        __syncthreads();

        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int threadNum = threadIdInMatrix; threadNum < matrixSize; threadNum += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, threadNum, matrixSize);
            }
            __syncthreads();
            
            for (int threadNum = threadIdInMatrix; threadNum < matrixSize; threadNum += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, threadNum, matrixSize);
            }
            __syncthreads();
        }

        for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
            inversion(sh_A, rowIdx, matrixSize);
        }
        __syncthreads();

        // printf("Inversion done\n");

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            int swl_idx = calc_swl_idx(k, matrixSize, numElements);
            A[swl_idx + mtrxOffset] = sh_A[swl_idx];
        }

    }


    __syncwarp();
}
