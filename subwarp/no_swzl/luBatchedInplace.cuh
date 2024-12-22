// #define _CG_ABI_EXPERIMENTAL

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

template<typename T>
__device__ __forceinline__ void comp_U(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix) {
    // if (colIdx < matrixSize) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx];
        }

        A[rowIdx * matrixSize + colIdx] = A[rowIdx * matrixSize + colIdx] - sum;
    // }

    // else {
    //     return;
    // }
}


template<typename T>
__device__ __forceinline__ void comp_L(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix) {
    // if (colIdx < matrixSize) {
        T sum = static_cast<T>(0.0);

        #pragma unroll 
        for (int l = 0; l < rowIdx; l++) {
            sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
    // }

    // else {
    //     return;
    // }

}


template<typename T>
__device__ __forceinline__ void inversion(T* A, int colIdx, int matrixSize) {
    // if (colIdx < matrixSize) {
        T y[32];

        for (int i = 0; i < matrixSize; i++) {
            y[i] = static_cast<T>(0.0);
        }

        // forward substitution
        #pragma unroll 
        for (int i = 0; i < matrixSize; i++) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = 0; j < i; j++) {
                sum += A[(i * matrixSize) + j] * y[j];
            }
            y[i] = (((i == colIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        #pragma unroll 
        for (int i = matrixSize-1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            // matrix_group.sync();
            #pragma unroll 
            for (int j = i+1; j < matrixSize; j++) {
                sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + colIdx];
            }
            A[(i * matrixSize) + colIdx] = (y[i] - sum) / A[(i * matrixSize) + i];
        }
    // }

    // else {
    //     return;
    // }

}


template<typename T, int matrixSize, int threadsPerMatrix, int matricesPerBlock, int numMatrices>
__global__ void batched_lu_subwarp(T* A) {

    int matrixIdInBlock =  threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdx.x % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[matrixIdInBlock * (numElements) + 1];

        // auto matrix_group = cg::tiled_partition<threadsPerMatrix>(cg::this_thread_block());
        // auto matrix_group = cg::experimental::tiled_partition<threadsPerMatrix>(cg::this_thread_block());

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[k] = A[k + mtrxOffset];
        }
        // matrix_group.sync();
        __syncthreads();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            #pragma unroll
            for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            }
            // comp_U(sh_A, rowIdx, (rowIdx+threadIdInMatrix), matrixSize, threadsPerMatrix);
            // matrix_group.sync();
            __syncthreads();

            #pragma unroll
            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            }
            // comp_L(sh_A, rowIdx, (rowIdx+1+threadIdInMatrix), matrixSize, threadsPerMatrix);
            // matrix_group.sync();
            __syncthreads();
        }

        #pragma unroll
        for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            inversion(sh_A, colIdx, matrixSize);
        }
        // inversion(sh_A, threadIdInMatrix, matrixSize);
        // matrix_group.sync();
        __syncthreads();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[k];
        }
    }
}
