#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


__device__ int calc_swl_idx_lei(int k, int matrixSize, int matrixIdInBlock, int numElements) {

    int row = k / matrixSize;
    int col = k % matrixSize;
    int x_swz = (col ^ row) + (matrixIdInBlock * numElements);

    if (int(k / matrixSize) > 0) {
        x_swz += matrixSize * (k / matrixSize);
    }

    // if (k % 7 == 0 && matrixIdInBlock == 0) {
    //     printf("oldIdx: %d newIdx: %d\n", k, x_swz);
    // }

    return x_swz;
}


template<typename T>
__device__ void comp_U_shfl(T* A, int rowIdx, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {

        for (int colIdx = rowIdx; colIdx < matrixSize; colIdx++) {
            T sum = static_cast<T>(0.0);

            for (int l = threadIdx.x % threadsPerMatrix; l < rowIdx; l += threadsPerMatrix) {
                // sum += A[calc_swl_idx_lei(rowIdx * matrixSize + colIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(colIdx * matrixSize + colIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
                sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx];
            }

            for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
            }

            if (threadIdx.x % threadsPerMatrix == 0) {
                A[rowIdx * matrixSize + colIdx] -= sum;
                // (&A[rowIdx * matrixSize + colIdx], -sum);
            }

            // print sum
            // if (threadIdx.x % threadsPerMatrix == 0) {
            //     printf("sum from U: %f %d %d %d\n", sum, threadIdx.x, rowIdx, colIdx);
            // }
        }

        // A[rowIdx * matrixSize + colIdx] = A[rowIdx * matrixSize + colIdx] - sum;
        // A[calc_swl_idx_lei(rowIdx * matrixSize + colIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = A[rowIdx * matrixSize + colIdx];
        // printf("A[%d][%d]: %f\n", rowIdx, colIdx, A[rowIdx * matrixSize + colIdx]);

}


template<typename T>
__device__ void comp_U_laneId(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
        T sum = static_cast<T>(0.0);
        int laneId = threadIdx.x % threadsPerMatrix;
        
        for (int l = laneId; l < rowIdx; l += threadsPerMatrix) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx];
        }
        
        for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
        }
        
        // if (laneId == 0) {
        if (laneId == threadIdx.x % threadsPerMatrix == 0) {
            A[rowIdx * matrixSize + colIdx] -= sum;
        }
}


template<typename T>
__device__ void comp_U(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
    // if (colIdx >= rowIdx) {

        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + colIdx]; 
        }

        A[rowIdx * matrixSize + colIdx] = A[rowIdx * matrixSize + colIdx] - sum;

                    // if (threadIdx.x % threadsPerMatrix == 0) {
                // printf("sum from U: %f %d %d %d\n", sum, threadIdx.x, rowIdx, colIdx);
            // }
        // A[calc_swl_idx_lei(rowIdx * matrixSize + colIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = A[rowIdx * matrixSize + colIdx];
        // printf("A[%d][%d]: %f\n", rowIdx, colIdx, A[rowIdx * matrixSize + colIdx]);
    // }

    // else {
    //     return;
    // }
}


template<typename T>
__device__ void comp_L_shfl(T* A, int rowIdx, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
 
        for (int colIdx = rowIdx+1; colIdx < matrixSize; colIdx++) {
            T sum = static_cast<T>(0.0);

            for (int l = threadIdx.x % threadsPerMatrix; l < rowIdx; l += threadsPerMatrix) {
                // sum += A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(colIdx * matrixSize + colIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
                sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
            }

            for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
            }

            if (threadIdx.x % threadsPerMatrix == 0) {
                A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
                // atomicAdd(&A[colIdx * matrixSize + rowIdx], -sum) / A[rowIdx * matrixSize + rowIdx];
            }
            // // print sum
            // if (threadIdx.x % threadsPerMatrix == 0) {
            //     printf("sum from L: %f\n", sum);
            // }
        }

        // A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
        // A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = A[colIdx * matrixSize + rowIdx];
        // printf("A[%d][%d]: %f\n", colIdx, rowIdx, A[colIdx * matrixSize + rowIdx]);

}


template<typename T>
__device__ void comp_L_laneId(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
        T sum = static_cast<T>(0.0);
        int laneId = threadIdx.x % threadsPerMatrix;
        
        for (int l = laneId; l < rowIdx; l += threadsPerMatrix) {
            sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
        }
        
        for (int offset = threadsPerMatrix/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset, threadsPerMatrix);
        }
        
        // if (laneId == 0) {
        if (laneId == threadIdx.x % threadsPerMatrix == 0) {
            A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
        }
}

template<typename T>
__device__ void comp_L(T* A, int rowIdx, int colIdx, int matrixSize, int threadsPerMatrix, int matrixIdInBlock) {
    // if (colIdx > rowIdx) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            // sum += A[calc_swl_idx_lei(colIdx * matrixSize + l, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(l * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
            sum += A[colIdx * matrixSize + l] * A[l * matrixSize + rowIdx];
        }

        A[colIdx * matrixSize + rowIdx] = (A[colIdx * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
        // A[calc_swl_idx_lei(colIdx * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = A[colIdx * matrixSize + rowIdx];
    //     // printf("A[%d][%d]: %f\n", colIdx, rowIdx, A[colIdx * matrixSize + rowIdx]);
    // }

    // else {
    //     return;
    // }
}


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize, int matrixIdInBlock) {
    // if (rowIdx < matrixSize) {
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
                // sum += A[calc_swl_idx_lei(i * matrixSize + j, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * y[j];
                sum += A[(i * matrixSize) + j] * y[j];
            }
            y[i] = (((i == rowIdx) ? static_cast<T>(1.0) : static_cast<T>(0.0)) - sum);
        }

        // backward substitution
        for (int i = matrixSize - 1; i >= 0; i--) {
            T sum = static_cast<T>(0.0);
            #pragma unroll
            for (int j = i + 1; j < matrixSize; j++) {
                // sum += A[calc_swl_idx_lei(i * matrixSize + j, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] * A[calc_swl_idx_lei(j * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)];
                sum += A[(i * matrixSize) + j] * A[(j * matrixSize) + rowIdx];
            }
            // A[calc_swl_idx_lei(i * matrixSize + rowIdx, matrixSize, matrixIdInBlock, matrixSize * matrixSize)] = (y[i] - sum) / A[(i * matrixSize) + i];
            A[(i * matrixSize) + rowIdx] = (y[i] - sum) / A[(i * matrixSize) + i]; 
        }
    // }

    // else {
    //     return;
    // }
}


template<typename T>
__global__ void batched_lu_subwarp(T* A, int matrixSize, int numMatrices, int threadsPerMatrix, int matricesPerBlock) {
    // int blockNum = blockIdx.x;
    // int threadIdInBlock = threadIdx.x;
    int matrixIdInBlock =  threadIdx.x / threadsPerMatrix;
    int globalMatrixId = (blockIdx.x * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdx.x % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[0];

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            // int swl_idx = calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements);
            // sh_A[swl_idx] = A[k + mtrxOffset];
            sh_A[k] = A[k + mtrxOffset];
            // int swl_idx = calc_swl_idx(k, matrixSize, numElements);
            // printf("swz_idx: %d, k: %d\n", swz_idx, k);
            // use lgds to avoid bank conflicts
            // sh_A[k] = (A[k + mtrxOffset]);
            // sh_A[k] = __ldg(&A[k + mtrxOffset]);
            // printf("sh_A[%d]: %f\n", k + bankOffset, sh_A[k + bankOffset]);
        }
        __syncthreads();

        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            // for (int colIdx = rowIdx+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {
            //     comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            // }

            // int colIdx = rowIdx + threadIdInMatrix;
            // while (colIdx < matrixSize) {
            //     comp_U(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            //     colIdx += threadsPerMatrix;
            // }

            comp_U_shfl(sh_A, rowIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);

            // for (int colIdx = rowIdx; colIdx < matrixSize; colIdx++) {
            //     comp_U_laneId(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            // }

            __syncthreads();

            // for (int colIdx = rowIdx+1+threadIdInMatrix; colIdx < matrixSize; colIdx += threadsPerMatrix) {            
            //     comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            // }

            // colIdx = rowIdx + 1 + threadIdInMatrix;
            // while (colIdx < matrixSize) {
            //     comp_L(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            //     colIdx += threadsPerMatrix;
            // }

            comp_L_shfl(sh_A, rowIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);

            // for (int colIdx = rowIdx+1; colIdx < matrixSize; colIdx++) {
            //     comp_L_laneId(sh_A, rowIdx, colIdx, matrixSize, threadsPerMatrix, matrixIdInBlock);
            // }

            __syncthreads();
        }

        for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
            inversion(sh_A, rowIdx, matrixSize, matrixIdInBlock);
        }
        __syncthreads();

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            // int swl_idx = calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements);
            // A[k + mtrxOffset] = sh_A[swl_idx];
            A[k + mtrxOffset] = sh_A[k];
            // asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(&A[k + mtrxOffset]) , "f"(sh_A[swl_idx]));
            // asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(&A[k + mtrxOffset]) , "f"(sh_A[k]));
            // A[k + mtrxOffset] = sh_A[k];
        }
    }
    __syncwarp();
}