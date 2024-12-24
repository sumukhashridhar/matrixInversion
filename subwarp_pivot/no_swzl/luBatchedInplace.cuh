// #define _CG_ABI_EXPERIMENTAL

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T>
__device__ __forceinline__ void swap_rows(T* A, int row1, int row2, int matrixSize) {
    #pragma unroll
    for (int j = 0; j < matrixSize; j++) {
        T temp = A[row1 * matrixSize + j];
        A[row1 * matrixSize + j] = A[row2 * matrixSize + j];
        A[row2 * matrixSize + j] = temp;
    }
}

template<typename T>
__device__ __forceinline__ int find_pivot(T* A, int k, int matrixSize) {
    int pivot_row = k;
    T max_val = abs(A[k * matrixSize + k]);
    
    #pragma unroll
    for (int i = k + 1; i < matrixSize; i++) {
        T val = abs(A[i * matrixSize + k]);
        if (val > max_val) {
            max_val = val;
            pivot_row = i;
        }
    }
    return pivot_row;
}

template<typename T>
__device__ __forceinline__ void comp_U(T* A, int* pivots, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix) {
    T sum = static_cast<T>(0.0);

    #pragma unroll
    for (int l = 0; l < rowIdx; l++) {
        sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx];
    }

    A[rowIdx * matrixSize + colIdx] = A[rowIdx * matrixSize + colIdx] - sum;
}

template<typename T>
__device__ __forceinline__ void comp_L(T* A, int* pivots, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix) {
    T sum = static_cast<T>(0.0);

    #pragma unroll 
    for (int l = 0; l < rowIdx; l++) {
        sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
    }

    A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
}

template<typename T>
__device__ __forceinline__ void comp_inv(T* A, int* pivots, int colIdx, int matrixSize) {
    T y[32];
    T x[32];

    // init arrays
    #pragma unroll
    for (int i = 0; i < matrixSize; i++) {
        y[i] = static_cast<T>(0.0);
        x[i] = static_cast<T>(0.0);
    }

    // forward substitution
    #pragma unroll 
    for (int i = 0; i < matrixSize; i++) {
        T b = (pivots[i] == colIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0);
        
        T sum = static_cast<T>(0.0);
        #pragma unroll
        for (int j = 0; j < i; j++) {
            sum += A[(i * matrixSize) + j] * y[j];
        }
        y[i] = b - sum;
    }

    // backward substitution
    #pragma unroll 
    for (int i = matrixSize-1; i >= 0; i--) {
        T sum = static_cast<T>(0.0);
        #pragma unroll 
        for (int j = i+1; j < matrixSize; j++) {
            sum += A[(i * matrixSize) + j] * x[j];
        }
        x[i] = (y[i] - sum) / A[(i * matrixSize) + i];
    }

    #pragma unroll
    for (int i = 0; i < matrixSize; i++) {
        A[(i * matrixSize) + colIdx] = x[i];
    }
}

template<typename T, int matrixSize, int threadsPerMatrix, int matricesPerBlock, int numMatrices>
__global__ void batched_lu_subwarp(T* A) {
    int matrixIdInBlock = threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + matrixIdInBlock;

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix = threadIdx.x % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T* sh_A = &shmem[matrixIdInBlock * (numElements + matrixSize) + 1];
        int* pivots = (int*)(&sh_A[numElements]);

        // init pivots
        if (threadIdInMatrix < matrixSize) {
            pivots[threadIdInMatrix] = threadIdInMatrix;
        }

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[k] = A[k + mtrxOffset];
        }
        __syncthreads();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            if (threadIdInMatrix == 0) {
                int pivot_row = find_pivot(sh_A, rowIdx, matrixSize);
                if (pivot_row != rowIdx) {
                    swap_rows(sh_A, rowIdx, pivot_row, matrixSize);
                    int temp = pivots[rowIdx];
                    pivots[rowIdx] = pivots[pivot_row];
                    pivots[pivot_row] = temp;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int colIdx = rowIdx + threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U(sh_A, pivots, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            }
            __syncthreads();

            #pragma unroll
            for (int colIdx = rowIdx + 1 + threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L(sh_A, pivots, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            }
            __syncthreads();
        }

        #pragma unroll
        for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            comp_inv(sh_A, pivots, colIdx, matrixSize);
        }
        __syncthreads();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[k];
        }
    }
}