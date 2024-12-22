#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include "verify.hpp"

namespace cg = cooperative_groups;


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T>
__device__ void comp_U_shfl(T* A, int rowIdx, int matrixSize, int threadsPerMatrix) {

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


template<typename T>
__device__ void comp_U(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx]; 
        }

        A[rowIdx * matrixSize + colIdx] = A[rowIdx * matrixSize + colIdx] - sum;

}


template<typename T>
__device__ void comp_L_shfl(T* A, int rowIdx, int matrixSize, int threadsPerMatrix) {
 
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


template<typename T>
__device__ void comp_L(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
}


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize) {
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
            sh_A[k] = A[k + mtrxOffset];
        }
        matrix_group.sync();

        #pragma unroll 8
        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            // #pragma unroll
            // for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            //     comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            // }
            // matrix_group.sync();

            comp_U_shfl(sh_A, rowIdx, matrixSize, threadsPerMatrix);

            // #pragma unroll
            // for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            //     comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix);
            // }
            // matrix_group.sync();

            comp_L_shfl(sh_A, rowIdx, matrixSize, threadsPerMatrix);
        }

        #pragma unroll
        for (int colIdx = threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            inversion(sh_A, colIdx, matrixSize);
        }
        matrix_group.sync();

        #pragma unroll
        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            A[k + mtrxOffset] = sh_A[k];
        }
    }
}