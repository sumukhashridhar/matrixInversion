#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "verify.hpp"


#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }


template<typename T>
__device__ void comp_U(T* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum >= rowIdx && threadNum < matrixSize) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[rowIdx * matrixSize + l] * A[l * matrixSize + threadNum]; 
        }
        A[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
        // printf("A[%d][%d]: %f\n", rowIdx, threadNum, A[rowIdx * matrixSize + threadNum]);
    }

    else {
        return;
    }
}


template<typename T>
__device__ void comp_L(T* A, int rowIdx, int threadNum, int matrixSize) {
    if (threadNum > rowIdx && threadNum < matrixSize) {
        T sum = static_cast<T>(0.0);

        #pragma unroll
        for (int l = 0; l < rowIdx; l++) {
            sum += A[threadNum * matrixSize + l] * A[l * matrixSize + rowIdx];
        }
        A[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / A[rowIdx * matrixSize + rowIdx];
        // printf("A[%d][%d]: %f\n", threadNum, rowIdx, A[threadNum * matrixSize + rowIdx]);
    }

    else {
        return;
    }
}


template<typename T>
__device__ void inversion(T* A, int rowIdx, int matrixSize) {
    if (rowIdx < matrixSize) {
        T y[128];

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

    else {
        return;
    }
}


// template<typename T>
__device__ int calc_swl_idx(int k, int matrixSize, int numElements) {
    // impl has div and mod, expnsve on GPU
    // int count = k / matrixSize;
    // printf("count is: %d\n", count);
    
    int newIdx = (k * matrixSize) % numElements + (k / matrixSize);

    // printf("oldIdx: %d newIdx: %d\n", k, newIdx);
    // int bankSize = 32;
    // int row = k / matrixSize;
    // int col = k % matrixSize;

    // int newIdx = (row * matrixSize + ((col + (row % bankSize)) % matrixSize)); 

    return newIdx;   
}

__device__ int calc_swl_idx_lei(int k, int matrixSize, int matrixIdInBlock, int numElements) {
    // int swl_size = matrixSize * sizeof(FpType);
    // int y_c = (k * sizeof(FpType)) / swl_size;
    // int x_c = (k * sizeof(FpType)) % swl_size;

    // int x_c_swl = y_c ^ x_c;
    // // x_swz = x_chunk_swz × sizeof(TC) / sizeof(T) % NX + x % (sizeof(TC) / sizeof(T))
    // int x_swz = (x_c_swl * sizeof(FpType) * matrixSize) / sizeof(FpType) % numElements + k % (sizeof(FpType) * matrixSize / sizeof(FpType));

    int row = k / matrixSize;
    int col = k % matrixSize;
    int x_swz = (col ^ row) + (matrixIdInBlock * numElements);

    if (int(k / matrixSize) > 0) {
        x_swz += matrixSize * (k / matrixSize);
    }

    // int x_swz = (row * matrixSize + ((col + (row % 32)) % matrixSize));
    if (k % 7 == 0 && matrixIdInBlock == 0) {
        // x_swz += matrixIdInBlock * matrixSize;
        printf("oldIdx: %d newIdx: %d\n", k, x_swz);
    }
    // printf("oldIdx: %d newIdx: %d\n", k, x_swz);

    return x_swz;
}


template<typename T>
__global__ void batched_lu_subwarp(T* A, int matrixSize, int numMatrices, int threadsPerMatrix, int matricesPerBlock) {
    int blockNum = blockIdx.x;
    int threadIdInBlock = threadIdx.x;
    int matrixIdInBlock =  threadIdInBlock / threadsPerMatrix;
    int globalMatrixId = (blockNum * matricesPerBlock) + (matrixIdInBlock);

    if (globalMatrixId < numMatrices) {
        int threadIdInMatrix =  threadIdInBlock % threadsPerMatrix;
        int numElements = matrixSize * matrixSize;
        int mtrxOffset = globalMatrixId * numElements;
        
        extern __shared__ T shmem[];
        T *sh_A = &shmem[(matrixIdInBlock) * (numElements+1)];

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            int swl_idx = calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements);
            sh_A[swl_idx] = A[k + mtrxOffset];
            // sh_A[k] = A[k + mtrxOffset];
            // int swl_idx = calc_swl_idx(k, matrixSize, numElements);
            // printf("swz_idx: %d, k: %d\n", swz_idx, k);
            // use lgds to avoid bank conflicts
            // sh_A[k] = (A[k + mtrxOffset]);
            // sh_A[k] = __ldg(&A[k + mtrxOffset]);
            // printf("sh_A[%d]: %f\n", k + bankOffset, sh_A[k + bankOffset]);
        }
        __syncthreads();

        for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
            for (int threadNum = threadIdInMatrix; threadNum < matrixSize; threadNum += threadsPerMatrix) {
                comp_U(sh_A, rowIdx, threadNum, matrixSize);
            }
            __syncthreads();
            
            for (int threadNum = threadIdInMatrix; threadNum < matrixSize; threadNum += threadsPerMatrix) {
                comp_L(sh_A, rowIdx, threadNum, matrixSize);
            }
            __syncthreads();
        }

        for (int rowIdx = threadIdInMatrix; rowIdx < matrixSize; rowIdx += threadsPerMatrix) {
            inversion(sh_A, rowIdx, matrixSize);
        }
        __syncthreads();

        for (int k = threadIdInMatrix; k < numElements; k += threadsPerMatrix) {
            int swl_idx = calc_swl_idx_lei(k, matrixSize, matrixIdInBlock, numElements);
            A[k + mtrxOffset] = sh_A[swl_idx];
            // A[k + mtrxOffset] = sh_A[k];
            // asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(&A[k + mtrxOffset]) , "f"(sh_A[swl_idx]));
            // asm volatile("st.global.cg.f32 [%0], %1;" : : "l"(&A[k + mtrxOffset]) , "f"(sh_A[k]));
            // A[k + mtrxOffset] = sh_A[k];
        }
    }
    __syncwarp();
}
