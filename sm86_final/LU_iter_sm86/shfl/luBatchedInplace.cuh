#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T, int matrixSize, int threadsPerMatrix>
__device__ void comp_U_shfl(T* __restrict__ A, int rowIdx) {

        for (int colIdx = rowIdx; colIdx < matrixSize; colIdx++) {
            T sum = static_cast<T>(0.0);

            for (int l = threadIdx.x % threadsPerMatrix; l < rowIdx; l += threadsPerMatrix) {
                sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx];
            }

            for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
            }

            if (threadIdx.x % threadsPerMatrix == 0) {
                A[rowIdx * matrixSize + colIdx] -= sum;
            }

        }

}


template<typename T, int matrixSize, int threadsPerMatrix>
__device__ void comp_L_shfl(T* __restrict__ A, int rowIdx) {
 
        for (int colIdx = rowIdx+1; colIdx < matrixSize; colIdx++) {
            T sum = static_cast<T>(0.0);

            for (int l = threadIdx.x % threadsPerMatrix; l < rowIdx; l += threadsPerMatrix) {
                sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
            }

            for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
            }

            if (threadIdx.x % threadsPerMatrix == 0) {
                A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
            }
        }
}


template<typename T, int matrixSize>
__device__ void inversion(T* __restrict__ A, int rowIdx) {
        T y[matrixSize];
        T x[matrixSize];

        // initialize y to 0
        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            y[i] = static_cast<T>(0.0);
            x[i] = static_cast<T>(0.0);
        }

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
                sum += A[(i * matrixSize) + j] * x[j];
            }
            x[i] = (y[i] - sum) / A[(i * matrixSize) + i];
        }
        __syncthreads();

        // copy back to A
        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            A[(i * matrixSize) + rowIdx] = x[i];
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
        T *sh_A = &shmem[matrixIdInBlock * numElements];

        if (threadIdInMatrix < threadsPerMatrix) {
            #pragma unroll
            for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
                sh_A[k] = A[k + mtrxOffset];
            }
        }
        __syncthreads();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            if (threadIdInMatrix < threadsPerMatrix) {
                comp_U_shfl<T, matrixSize, threadsPerMatrix>(sh_A, rowIdx);
            }
            __syncthreads();

            if (threadIdInMatrix < threadsPerMatrix) {
                comp_L_shfl<T, matrixSize, threadsPerMatrix>(sh_A, rowIdx);
            }
            __syncthreads();
        }

        // if (threadIdInMatrix < threadsPerMatrix) {
        //     #pragma unroll
        //     for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
        //         inversion<T, matrixSize>(sh_A, colIdx);
        //     }
        // }
        // __syncthreads();

        if (threadIdInMatrix < threadsPerMatrix) {
            #pragma unroll
            for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
                A[k + mtrxOffset] = sh_A[k];
            }
        }
    }
}