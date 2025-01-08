#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


__device__ __forceinline__ int calc_swl_idx_lei(int k, int matrixSize, int swzOffset) {
    // Calculate matrixSizeLog2
    int matrixSizeLog2 = 31 - __clz(matrixSize);
    
    int row = k >> matrixSizeLog2;
    int col = k & (matrixSize - 1);

    return ((col ^ row) + (row << matrixSizeLog2)) + swzOffset;
}


template<typename T, int matrixSize, int threadsPerMatrix>
__device__ __forceinline__ void comp_U(T* __restrict__ A, int rowIdx, int threadNum, int swzOffset) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + threadNum, matrixSize, swzOffset)];
        }

        A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, swzOffset)] = A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, swzOffset)] - sum;
}


template<typename T, int matrixSize, int threadsPerMatrix>
__device__ __forceinline__ void comp_L(T* __restrict__ A, int rowIdx, int threadNum, int swzOffset) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(threadNum * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, swzOffset)];
        }

        A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, swzOffset)] = (A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, swzOffset)] - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize, swzOffset)];
}


template<typename T, int matrixSize>
__device__ __forceinline__ void inversion(T* __restrict__ A, int colIdx, int swzOffset) {
        T y[matrixSize];
        T x[matrixSize];

        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            y[i] = static_cast<T>(0.0);
            x[i] = static_cast<T>(0.0);
        }

        // forward substitution
        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = 0; j < i; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, swzOffset)] * y[j];
            }
            y[i] = (((i == colIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        #pragma unroll
        for (int i = matrixSize-1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i+1; j < matrixSize; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, swzOffset)] * x[j];
            }
            x[i] = (y[i] - sum) / A[calc_swl_idx_lei((i * matrixSize) + i, matrixSize, swzOffset)];
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            A[calc_swl_idx_lei((i * matrixSize) + colIdx, matrixSize, swzOffset)] = x[i];
        }
}


template<typename T, int matrixSize, int threadsPerMatrix, int matricesPerBlock, int numMatrices>
__launch_bounds__(BLOCK_SIZE)
__global__ void batched_lu_subwarp(T* __restrict__ A) {

    int matrixIdInBlock =  threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdx.x % threadsPerMatrix;
        constexpr int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[matrixIdInBlock * (numElements)];

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[calc_swl_idx_lei(k, matrixSize, (matrixIdInBlock * numElements))] = A[k + mtrxOffset];
        }
        __syncthreads();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            #pragma unroll
            for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U<T, matrixSize, threadsPerMatrix>(sh_A, rowIdx, colIdx, (matrixIdInBlock * numElements));
            }
            __syncthreads();

            #pragma unroll
            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L<T, matrixSize, threadsPerMatrix>(sh_A, rowIdx, colIdx, (matrixIdInBlock * numElements));
            }
            __syncthreads();
        }

        // #pragma unroll
        // for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
        //     inversion<T, matrixSize>(sh_A, colIdx, (matrixIdInBlock * numElements));
        // }
        // __syncthreads();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[calc_swl_idx_lei(k, matrixSize, (matrixIdInBlock * numElements))];
        }
    }
}
