#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


__device__ int calc_swl_idx_lei(int k, int matrixSize, int matrixIdInBlock, int numElements) {

    int row = k / matrixSize;
    int col = k % matrixSize;
    int x_swz = (col ^ row); // + (matrixIdInBlock * numElements);

    if (int(k / matrixSize) > 0) {
        x_swz += matrixSize * (k / matrixSize);
    }

    x_swz += matrixIdInBlock * numElements;

    // if (k % 7 == 0 && matrixIdInBlock == 0) {
    //     printf("oldIdx: %d newIdx: %d\n", k, x_swz);
    // }

    return x_swz;
}


template<typename T>
__device__ void comp_U(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        T sum = static_cast<T>(0.0);
        // unsigned int mask = (1 << threadsPerMatrix) - 1;

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + threadNum, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
            // sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }

        // A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
        A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] - sum;
        // printf("A[%d][%d]: %f\n", rowIdx, threadNum, A[rowIdx * matrixSize + threadNum]);
    }

    else {
        return;
    }
}


template<typename T>
__device__ void comp_L(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
    if (threadNum > rowIdx && threadNum < matrixSize) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(threadNum * matrixSize + l, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
            // sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        // A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
        A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = (A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
        // printf("A[%d][%d]: %f\n", threadNum, rowIdx, A[threadNum * matrixSize + rowIdx]);
    }

    else {
        return;
    }
}


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize, int matrixIdInBlock) {
    if (rowIdx < matrixSize) {
        T y[32];

        // forward substitution
        for (int i = 0; i < matrixSize; i++) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = 0; j < i; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * y[j];
                // sum += A[(i * matrixSize) + j] * y[j];
            }
            y[i] = (((i == rowIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        for (int i = matrixSize - 1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei((j * matrixSize) + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
                // sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + rowIdx];
            }
            A[calc_swl_idx_lei((i * matrixSize) + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = (y[i] - sum) / A[calc_swl_idx_lei((i * matrixSize) + i, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
            // A[(i * matrixSize) + rowIdx] = (y[i] - sum) / A[(i * matrixSize) + i]; 
        }
    }

    else {
        return;
    }
}


template<typename T>
__global__ void batched_lu_subwarp(T* A, int matrixSize, int numMatrices, int threadsPerMatrix, int matricesPerBlock) {
    int matrixIdInBlock =  threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdx.x % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[0];

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements)] = A[k + mtrxOffset];
            // sh_A[k] = A[k + mtrxOffset];
        }
        __syncthreads();

        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int threadNum = threadIdInMatrix; threadNum < matrixSize; threadNum += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, threadNum, matrixSize, threadsPerMatrix, matrixIdInBlock);
            }
            __syncthreads();
            
            for (int threadNum = threadIdInMatrix; threadNum < matrixSize; threadNum += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, threadNum, matrixSize, threadsPerMatrix, matrixIdInBlock);
            }
            __syncthreads();
        }

        for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
            inversion(sh_A, rowIdx, matrixSize, matrixIdInBlock);
        }
        __syncthreads();

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements)];
            // A[k + mtrxOffset] = sh_A[k];
        }
    }
    __syncwarp();
}