#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T>
__device__ void comp_U(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }

        A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;

}


template<typename T>
__device__ void comp_L(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
}


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize, int matrixIdInBlock) {
        T y[32];

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


template<typename T>//, int threadsPerMatrix=8, int matricesPerBlock=4>
__global__ void batched_lu_subwarp(T* A, int matrixSize, int numMatrices, int threadsPerMatrix, int matricesPerBlock) {
    int matrixIdInBlock =  threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdx.x % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[0];
        // T *sh_A = &shmem[matrixIdInBlock * (numElements+1)];

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[k] = A[k + mtrxOffset];
        }
        __syncthreads();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            }

            __syncthreads();

            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            }
            
            __syncthreads();
        }

        for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
            inversion(sh_A, rowIdx, matrixSize, matrixIdInBlock);
        }
        __syncthreads();

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[k];
        }
    }
}