#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


__device__ __forceinline__ int calc_swl_idx_lei(int k, int matrixSize, int matrixIdInBlock, int numElements) {

    int row = k / matrixSize;
    int col = k % matrixSize;
    int x_swz = (col ^ row); 

    if (int(k / matrixSize) > 0) {
        x_swz += matrixSize * (k / matrixSize);
    }

    x_swz += matrixIdInBlock * numElements;

    return x_swz;
}


template<typename T>
__device__ __forceinline__ void comp_U(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + threadNum, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
        }

        A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] - sum;
}


template<typename T>
__device__ __forceinline__ void comp_L(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(threadNum * matrixSize + l, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
        }

        A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = (A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
}


template<typename T>
__device__ __forceinline__ void inversion(T* A, int rowIdx, int matrixSize, int matrixIdInBlock) {
        T y[32];

        // forward substitution
        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = 0; j < i; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * y[j];
            }
            y[i] = (((i == rowIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        #pragma unroll
        for (int i = matrixSize - 1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei((j * matrixSize) + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
            }
            A[calc_swl_idx_lei((i * matrixSize) + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = (y[i] - sum) / A[calc_swl_idx_lei((i * matrixSize) + i, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
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

        cg::thread_block cta = cg::this_thread_block();
        cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements)] = A[k + mtrxOffset];
            // sh_A[k] = A[k + mtrxOffset];
        }
        tile.sync();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            }

            tile.sync();

            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            }
            
            tile.sync();
        }

        for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
            inversion(sh_A, rowIdx, matrixSize, matrixIdInBlock);
        }
        tile.sync();

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements)];
            // A[k + mtrxOffset] = sh_A[k];
        }
    }
}