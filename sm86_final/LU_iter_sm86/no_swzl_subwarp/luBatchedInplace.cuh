#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

template<typename T, int matrixSize, int threadsPerMatrix>
__device__ __forceinline__ void comp_U(T* __restrict__ A, int rowIdx, int colIdx) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx];
        }

        A[rowIdx * matrixSize + colIdx] = A[rowIdx * matrixSize + colIdx] - sum;
}


template<typename T, int matrixSize, int threadsPerMatrix>
__device__ __forceinline__ void comp_L(T* __restrict__ A, int rowIdx, int colIdx) {
        T sum = static_cast<T>(0.0);

        #pragma unroll 
        for (int l = 0; l < rowIdx; l++) {
            sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
}


template<typename T, int matrixSize>
__device__ __forceinline__ void inversion(T* __restrict__ A, int colIdx) {
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
                sum += A[(i * matrixSize) + j] * y[j];
            }
            y[i] = (((i == colIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
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
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            A[(i * matrixSize) + colIdx] = x[i];
        }
}


template<typename T, int matrixSize, int threadsPerMatrix, int matricesPerBlock, int numMatrices>
__global__ void
__launch_bounds__(BLOCK_SIZE)
batched_lu_subwarp(T* __restrict__ A) {

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
            sh_A[k] = A[k + mtrxOffset];
        }
        __syncthreads();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            #pragma unroll
            for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U<T, matrixSize, threadsPerMatrix>(sh_A, rowIdx, colIdx);
            }
            __syncthreads();

            #pragma unroll
            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L<T, matrixSize, threadsPerMatrix>(sh_A, rowIdx, colIdx);
            }
            __syncthreads();
        }

        // #pragma unroll
        // for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
        //     inversion<T, matrixSize>(sh_A, colIdx);
        // }
        // __syncthreads();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[k];
        }
    }
}
