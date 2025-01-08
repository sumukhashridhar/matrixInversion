#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"

#define BLOCK_SIZE 256
#define MAX_MATRIX_SIZE 32

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T>
__device__ void comp_U(T* __restrict__ A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }

        A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;

}


template<typename T>
__device__ void comp_L(T* __restrict__ A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
}


template<typename T>
__device__ void inversion(T* __restrict__ A, int rowIdx, int matrixSize) {
        T y[MAX_MATRIX_SIZE];
        T x[MAX_MATRIX_SIZE];

        // init x and y
        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            x[i] = static_cast<T>(0.0);
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
            y[i] = (((i == rowIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        #pragma unroll
        for (int i = matrixSize - 1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                sum += A[(i * matrixSize) + j] * x[j];
            }
            x[i] = (y[i] - sum) / A[(i * matrixSize) + i];
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            A[(i * matrixSize) + rowIdx] = x[i];
        }
}


template<typename T>
__global__ void
__launch_bounds__(BLOCK_SIZE)
batched_lu_subwarp(T* __restrict__ A, int matrixSize, int numMatrices, int threadsPerMatrix, int matricesPerBlock) {
    int matrixIdInBlock =  threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdx.x % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
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
                comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            }

            __syncthreads();

            #pragma unroll
            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            }
            
            __syncthreads();
        }

        // #pragma unroll
        // for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
        //     inversion(sh_A, rowIdx, matrixSize);
        // }
        // __syncthreads();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[k];
        }
    }
}