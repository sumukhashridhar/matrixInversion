#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


__device__ __forceinline__ int calc_swl_idx_lei(int k, int matrixSize, int swzOffset) {
    // Calculate matrixSizeLog2
    int matrixSizeLog2 = 31 - __clz(matrixSize);
    
    int row = k >> matrixSizeLog2;
    int col = k & (matrixSize - 1);

    return ((col ^ row) + (row << matrixSizeLog2)) + swzOffset;
}


template<typename T>
__device__ void comp_U_shfl(T* A, int rowIdx, int matrixSize, int threadsPerMatrix, int swzOffset) {

        for (int colIdx = rowIdx; colIdx < matrixSize; colIdx++) {
            T sum = static_cast<T>(0.0);

            for (int l = threadIdx.x % threadsPerMatrix; l < rowIdx; l += threadsPerMatrix) {
                sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + colIdx, matrixSize, swzOffset)];
            }

            for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
            }

            if (threadIdx.x % threadsPerMatrix == 0) {
                A[calc_swl_idx_lei(rowIdx * matrixSize + colIdx, matrixSize, swzOffset)] -= sum;
            }
        }
}


template<typename T>
__device__ __forceinline__ void comp_U(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix, int swzOffset) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(rowIdx * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + colIdx, matrixSize, swzOffset)];
        }

        A[calc_swl_idx_lei(rowIdx * matrixSize + colIdx, matrixSize, swzOffset)] = A[calc_swl_idx_lei(rowIdx * matrixSize + colIdx, matrixSize, swzOffset)] - sum;
}


template<typename T>
__device__ void comp_L_shfl(T* A, int rowIdx, int matrixSize, int threadsPerMatrix, int swzOffset) {
 
        for (int colIdx = rowIdx+1; colIdx < matrixSize; colIdx++) {
            T sum = static_cast<T>(0.0);

            for (int l = threadIdx.x % threadsPerMatrix; l < rowIdx; l += threadsPerMatrix) {
                sum += A[calc_swl_idx_lei(colIdx * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, swzOffset)];
            }

            for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
            }

            if (threadIdx.x % threadsPerMatrix == 0) {
                A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, swzOffset)] = (A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, swzOffset)]  - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize, swzOffset)] ;
            }
        }
}


template<typename T>
__device__ __forceinline__ void comp_L(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix, int swzOffset) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[calc_swl_idx_lei(colIdx * matrixSize + l, matrixSize, swzOffset)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, swzOffset)];
        }

        A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, swzOffset)] = (A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, swzOffset)] - sum) / A[calc_swl_idx_lei(rowIdx * matrixSize + rowIdx, matrixSize, swzOffset)];
}


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize, int swzOffset) {
        T y[32];
        // initialize y to 0
        #pragma unroll
        for (int i = 0; i < matrixSize; i++) {
            y[i] = static_cast<T>(0.0);
        }

        // forward substitution
        for (int i = 0; i < matrixSize; i++) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = 0; j < i; j++) {
                // sum += A[(i * matrixSize) + j] * y[j];
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, swzOffset)] * y[j];
            }
            y[i] = (((i == rowIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        for (int i = matrixSize - 1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                // sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + rowIdx];
                sum += A[calc_swl_idx_lei((i * matrixSize) + j, matrixSize, swzOffset)] * A[calc_swl_idx_lei((j * matrixSize) + rowIdx, matrixSize, swzOffset)];
            }
            // A[(i * matrixSize) + rowIdx] = (y[i] - sum) / A[(i * matrixSize) + i]; 
            A[calc_swl_idx_lei((i * matrixSize) + rowIdx, matrixSize, swzOffset)] = (y[i] - sum) / A[calc_swl_idx_lei((i * matrixSize) + i, matrixSize, swzOffset)];
    
        }
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
        T *sh_A = &shmem[matrixIdInBlock * numElements];

        auto matrix_group = cg::tiled_partition<threadsPerMatrix>(cg::this_thread_block());
        // auto matrix_group = cg::experimental::tiled_partition<threadsPerMatrix>(cg::this_thread_block());

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            sh_A[calc_swl_idx_lei(k, matrixSize, (matrixIdInBlock * numElements))] = A[k + mtrxOffset];
            // sh_A[k] = A[k + mtrxOffset];
        }
        matrix_group.sync();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            // #pragma unroll
            // for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            //     comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, (matrixIdInBlock * numElements));
            // }
            // matrix_group.sync();

            comp_U_shfl(sh_A, rowIdx, matrixSize, threadsPerMatrix, (matrixIdInBlock * numElements));

            // #pragma unroll
            // for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            //     comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, (matrixIdInBlock * numElements));
            // }
            // matrix_group.sync();

            comp_L_shfl(sh_A, rowIdx, matrixSize, threadsPerMatrix, (matrixIdInBlock * numElements));
        }

        #pragma unroll
        for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            inversion(sh_A, colIdx, matrixSize, (matrixIdInBlock * numElements));
        }
        matrix_group.sync();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[calc_swl_idx_lei(k, matrixSize, (matrixIdInBlock * numElements))];
            // A[k + mtrxOffset] = sh_A[k];
        }
    }
}