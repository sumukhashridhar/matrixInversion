#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


// __device__ __forceinline__ int calc_swl_idx_lei(int k, int matrixSize, int swzOffset) {

//     int row = k / matrixSize;
//     // int col = k % matrixSize;
//     // int col = k - row * matrixSize;
//     // int x_swz = (col ^ row);
//     // int x_swz = (col ^ row) + (row * matrixSize) + swzOffset;
//     int x_swz = ((k - row * matrixSize) ^ row) + (row * matrixSize) + swzOffset;

//     return x_swz;
// }


__device__ __forceinline__ int calc_swl_idx_lei(int k, int matrixSize, int swzOffset) {
    // Calculate matrixSizeLog2
    int matrixSizeLog2 = 31 - __clz(matrixSize);
    
    int row = k >> matrixSizeLog2;
    int col = k & (matrixSize - 1);

    // if (threadIdx.x == 0) {
    //     printf("oldIdx: %d, newIdx: %d\n", k, ((col ^ row) | (row << matrixSizeLog2)) + swzOffset);
    // }

    return ((col ^ row) + (row << matrixSizeLog2)) + swzOffset;
}


template<typename T>
__device__ __forceinline__ void comp_U(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int swzOffset) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + threadNum, matrixSize, swzOffset)];
        }

        A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, swzOffset)] = A[calc_swl_idx_lei(rowIdx * matrixSize + threadNum, matrixSize, swzOffset)] - sum;
}


template<typename T>
__device__ __forceinline__ void comp_L(T* A, int rowIdx, int threadNum, int matrixSize, int threadsPerMatrix, int swzOffset) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(threadNum * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, swzOffset)];
        }

        A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, swzOffset)] = (A[calc_swl_idx_lei(threadNum * matrixSize + rowIdx, matrixSize, swzOffset)] - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize, swzOffset)];
}


template<typename T>
__device__ __forceinline__ void inversion(T* A, int colIdx, int matrixSize, int swzOffset) {
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
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, swzOffset)] * y[j];
            }
            // printf("sum from forward: %f, i: %d, colIdx: %d\n", sum, i, colIdx);
            y[i] = (((i == colIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // print y
        // for (int i = 0; i < matrixSize; i++) {
        //     printf("%f ", y[i]);
        // }

        // backward substitution
        #pragma unroll
        for (int i = matrixSize-1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i+1; j < matrixSize; j++) {
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, swzOffset)] * A[calc_swl_idx_lei((j * matrixSize) + colIdx, matrixSize, swzOffset)];
            }
            // print sum
            // printf("sum from backward: %f, i: %d, colIdx: %d\n", sum, i, colIdx);
            A[calc_swl_idx_lei((i * matrixSize) + colIdx, matrixSize, swzOffset)] = (y[i] - sum) / A[calc_swl_idx_lei((i * matrixSize) + i, matrixSize, swzOffset)];
    
        }
}


template<typename T, int matrixSize=32, int threadsPerMatrix, int matricesPerBlock, int numMatrices=1000>
__global__ void batched_lu_subwarp(T* A) {
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
            sh_A[calc_swl_idx_lei(k, matrixSize, (matrixIdInBlock * numElements))] = A[k + mtrxOffset];
            // sh_A[k] = A[k + mtrxOffset];
        }
        tile.sync();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, (matrixIdInBlock * numElements));
            }

            tile.sync();

            for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, (matrixIdInBlock * numElements));
            }
            
            tile.sync();
        }

        for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            inversion(sh_A, colIdx, matrixSize, (matrixIdInBlock * numElements));
            // __syncthread();
        }
        tile.sync();

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[calc_swl_idx_lei(k, matrixSize, (matrixIdInBlock * numElements))];
            // A[k + mtrxOffset] = sh_A[k];
        }
    }
}