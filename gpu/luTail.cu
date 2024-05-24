// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// typedef double FpType;

// __device__ void comp_U(FpType* A, FpType* L, FpType* U, int rowIdx, int threadNum, int matrixSize) {
//     if (threadNum >= rowIdx && threadNum < matrixSize) {
//     int l=0;
//     FpType sum = 0.0;

//     #pragma unroll
//     for (l = 0; l < rowIdx; l++) {
//         sum += L[rowIdx * matrixSize + l] * U[l * matrixSize + threadNum];
//     }
//     U[rowIdx * matrixSize + threadNum] = A[rowIdx * matrixSize + threadNum] - sum;
//     }

//     else {
//         return;
//     }
// }

// __device__ void comp_L(FpType* A, FpType* L, FpType* U, int rowIdx, int threadNum, int matrixSize) {
//     if (threadNum >= rowIdx && threadNum < matrixSize) {
//     int l=0;
//     FpType sum = 0.0;

//     if (rowIdx==threadNum) {
//         L[threadNum * matrixSize + threadNum] = 1.0;
//     }
//     else {
//         #pragma unroll
//         for (l = 0; l < rowIdx; l++) {
//             sum += L[threadNum * matrixSize + l] * U[l * matrixSize + rowIdx];
//         }
//         L[threadNum * matrixSize + rowIdx] = (A[threadNum * matrixSize + rowIdx] - sum) / U[rowIdx * matrixSize + rowIdx];
//     }
//     }

//     else {
//         return;
//     }
// }

// __device__ void forward_sub(FpType* L, FpType* e, FpType* y, int matrixSize) {
//     for (int i = 0; i < matrixSize; i++) {
//         FpType sum = 0.0;
//         #pragma unroll
//         for (int j = 0; j < i; j++) {
//             sum += L[i * matrixSize + j] * y[j];
//         }
//         y[i] = (e[i] - sum) / L[i * matrixSize + i];
//     }
// }

// __device__ void backward_sub(FpType* U, FpType* y, FpType* A_inv, int matrixSize) {
//     for (int i = matrixSize - 1; i >= 0; i--) {
//         FpType sum = 0.0;
//         #pragma unroll
//         for (int j = i + 1; j < matrixSize; j++) {
//             sum += U[i * matrixSize + j] * A_inv[j];
//         }
//         A_inv[i] = (y[i] - sum) / U[i * matrixSize + i];
//     }
// }

// __global__ void lu_decomp(FpType* A, FpType* L, FpType* U, FpType* e, FpType* y, FpType* A_inv, int matrixSize) {
    
//     for (int rowIdx = 0; rowIdx < matrixSize; rowIdx++) {
//         for (int j = threadIdx.x; j < matrixSize; j += blockDim.x) {
//             comp_U(A, L, U, rowIdx, j, matrixSize);
//         }

//         __syncthreads();

//         for (int j = threadIdx.x; j < matrixSize; j += blockDim.x) {
//             comp_L(A, L, U, rowIdx, j, matrixSize);
//         }

//         __syncthreads();
//     }

//     for (int rowIdx = threadIdx.x; rowIdx < matrixSize; rowIdx =+ blockDim.x) {
//         // e[rowIdx * matrixSize + rowIdx] = 1.0;
//         forward_sub(L, &e[rowIdx * matrixSize], &y[rowIdx * matrixSize], matrixSize);
//         backward_sub(U, &y[rowIdx * matrixSize], &A_inv[rowIdx * matrixSize], matrixSize);
//     }

//     __syncthreads();
// }

#include "luTail.cuh"

int main() {
    FpType *A, *L, *U, *LU, *e, *y, *A_inv;
    FpType *d_A, *d_L, *d_U, *d_e, *d_y, *d_A_inv;
    int matrixSize, numThreads;
    // let user input matrixSize and numThreads
    printf("Enter matrix size: ");
    scanf("%d", &matrixSize);
    printf("Enter number of threads: ");
    scanf("%d", &numThreads);
    int i=0, j=0;
    int numElements = matrixSize * matrixSize;
    FpType startT=0.0, endT=0.0;

    A = (FpType *)malloc(numElements * sizeof(FpType));
    L = (FpType *)malloc(numElements * sizeof(FpType));
    U = (FpType *)malloc(numElements * sizeof(FpType));
    LU = (FpType *)malloc(numElements * sizeof(FpType));
    e = (FpType *)malloc(numElements * sizeof(FpType));
    y = (FpType *)malloc(numElements * sizeof(FpType));
    A_inv = (FpType *)malloc(numElements * sizeof(FpType));

    cudaMalloc(&d_A, numElements * sizeof(FpType));
    cudaMalloc(&d_L, numElements * sizeof(FpType));
    cudaMalloc(&d_U, numElements * sizeof(FpType));
    cudaMalloc(&d_e, numElements * sizeof(FpType));
    cudaMalloc(&d_y, numElements * sizeof(FpType));
    cudaMalloc(&d_A_inv, numElements * sizeof(FpType));

    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    FILE *file = fopen("matrix.txt", "r");

    srand(time(NULL));
    for (i = 0; i < matrixSize; i++) {
        for (j = 0; j < matrixSize; j++) {
            fscanf(file, "%lf", &A[i * matrixSize + j]);
            // A[i * matrixSize + j] = inputMatrix[i * matrixSize + j];
            // A[i * matrixSize + j] = rand() % 10 + 1;
            L[i * matrixSize + j] = 0.0;
            U[i * matrixSize + j] = 0.0;
            LU[i * matrixSize + j] = 0.0;
            if (i == j) {
                e[i * matrixSize + j] = 1.0;
            }
            else {
                e[i * matrixSize + j] = 0.0;
            }
            y[i * matrixSize + j] = 0.0;
            A_inv[i * matrixSize + j] = 0.0;
        }
    }

    cudaMemcpy(d_A, A, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_inv, A_inv, numElements * sizeof(FpType), cudaMemcpyHostToDevice);

    startT = clock();
    lu_decomp<<<1, numThreads>>>(d_A, d_L, d_U, d_e, d_y, d_A_inv, matrixSize);
    endT = clock();

    cudaMemcpy(A, d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_inv, d_A_inv, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);

    printf("Time taken: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    // // print A
    // printf("Orig A:\n");
    // for (i = 0; i < matrixSize; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", A[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }

    // // print U
    // printf("U:\n");
    // for (i = 0; i < matrixSize; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", U[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }

    // // print L
    // printf("L:\n");
    // for (i = 0; i < matrixSize; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", L[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }

    // multipy L and U to check if A = LU
    for (i = 0; i < matrixSize; i++) {
        for (j = 0; j < matrixSize; j++) {
            FpType sum = 0.0;
            for (int k = 0; k < matrixSize; k++) {
                sum += L[i * matrixSize + k] * U[k * matrixSize + j];
            }
            LU[i * matrixSize + j] = sum;
        }
    }

    FpType diff=0.0;

    // subtract A from LU
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            diff += fabs(LU[i * matrixSize + j] - A[i * matrixSize + j]);
        }
    }

    if (isnan(diff))  {
        printf("Diff is nan\n");
    }

    if (fabs(diff) < 10e-6) {
        printf("LU decomposition is correct\n");
    }
    if (fabs(diff) > 10e-6) {
        printf("LU decomposition is incorrect\n");
    }

    printf("Diff is %f\n", diff);

    FpType result=0.0;
    FpType Id[matrixSize * matrixSize];

    // check if A_inv * A = I
    printf("Multiplication of A and A_inv:\n");
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            result = 0.0;
            for (int k = 0; k < matrixSize; k++) {
                result += A[i * matrixSize + k] * A_inv[j * matrixSize + k];
            }
            Id[j * matrixSize + i] = result;
            // printf("%f ", result);
        }
    // printf("\n");
    }

    // check if the multiplication is correct
    int idCnt = 0;
    for (int i = 0; i < matrixSize; i++) {
        #pragma unroll
        for (int j = 0; j < matrixSize; j++) {
            if (Id[j * matrixSize + i] == fabs(1.0)) {
                idCnt++;
            }
        }
    }

    if (idCnt == matrixSize) {
        printf("Count is %d\n", idCnt);
        printf("Inverse is correct\n");
    } else {
        printf("Count is %d\n", idCnt);
        printf("Inverse is incorrect\n");
    }

    // print A_inv
    printf("A_inv:\n");
    for (i = 0; i < matrixSize; i++) {
        for (j = 0; j < matrixSize; j++) {
            printf("%f ", Id[j * matrixSize + i]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_e);
    cudaFree(d_y);
    cudaFree(d_A_inv);

    free(A);
    free(L);
    free(U);
    free(LU);
    free(e);
    free(y);
    free(A_inv);

    return 0;
}
