// #define _CG_ABI_EXPERIMENTAL

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T>
__device__ __forceinline__ void find_pivot_parallel(T* A, int k, int matrixSize, int threadIdInMatrix, 
    int threadsPerMatrix, T* local_max_vals, int* local_max_indices) {
    int thread_max_idx = k;
    T val = static_cast<T>(0.0), thread_max_val = static_cast<T>(0.0);
    bool is_larger = false;
    
    for (int i = k + 1 + threadIdInMatrix; i < matrixSize; i += threadsPerMatrix) {
        val = abs(A[i * matrixSize + k]);
        is_larger = (val > thread_max_val);
        thread_max_val = is_larger ? val : thread_max_val;
        thread_max_idx = is_larger ? i : thread_max_idx;
    }
    
    local_max_vals[threadIdInMatrix] = thread_max_val;
    local_max_indices[threadIdInMatrix] = thread_max_idx;
    __syncthreads();
    
    for (int stride = threadsPerMatrix/2; stride > 0; stride >>= 1) {
        if (threadIdInMatrix < stride) {
            if (local_max_vals[threadIdInMatrix] < local_max_vals[threadIdInMatrix + stride]) {
                local_max_vals[threadIdInMatrix] = local_max_vals[threadIdInMatrix + stride];
                local_max_indices[threadIdInMatrix] = local_max_indices[threadIdInMatrix + stride];
            }
        }
        __syncthreads();
    }
}

template<typename T>
__device__ __forceinline__ void row_swap_parallel(T* A, int row1, int row2, int matrixSize, 
    int threadIdInMatrix, int threadsPerMatrix) {
    T temp = static_cast<T>(0.0);

    for (int j = threadIdInMatrix; j < matrixSize; j += threadsPerMatrix) {
        temp = A[row1 * matrixSize + j];
        A[row1 * matrixSize + j] = A[row2 * matrixSize + j];
        A[row2 * matrixSize + j] = temp;
    }
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

    // init arrays
    #pragma unroll
    for (int i = 0; i < matrixSize; i++) {
        y[i] = static_cast<T>(0.0);
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
            sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + colIdx];
        }
        A[(i * matrixSize) + colIdx] = (y[i] - sum) / A[(i * matrixSize) + i];
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
        T* sh_A = &shmem[matrixIdInBlock * (numElements + matrixSize + 2 * threadsPerMatrix)];
        int* pivots = (int*)(&sh_A[numElements]);

        T* local_max_vals = (T*)(&pivots[matrixSize]);
        int* local_max_indices = (int*)(&local_max_vals[threadsPerMatrix]);

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

            find_pivot_parallel(sh_A, rowIdx, matrixSize, threadIdInMatrix, 
                threadsPerMatrix, local_max_vals, local_max_indices);
            
            if (threadIdInMatrix == 0) {
                int pivot_row = local_max_indices[0];
                if (pivot_row != rowIdx) {
                    int temp = pivots[rowIdx];
                    pivots[rowIdx] = pivots[pivot_row];
                    pivots[pivot_row] = temp;
                }
            }
            __syncthreads();
            
            if (local_max_indices[0] != rowIdx) {
                row_swap_parallel(sh_A, rowIdx, local_max_indices[0], matrixSize, 
                    threadIdInMatrix, threadsPerMatrix);
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