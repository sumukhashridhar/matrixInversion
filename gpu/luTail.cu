#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void comp_U(double* A, double* L, double* U, int i, int j, int n) {
    if (j >= i && j < n) {
    int l=0;
    double sum = 0.0;

    // if (j==31)
    // printf("i: %d, j: %d, n: %d\n", i, j, n);

    #pragma unroll
    for (l = 0; l < i; l++) {
        sum += L[i * n + l] * U[l * n + j];
    }
    U[i * n + j] = A[i * n + j] - sum;
    }

    // else {
    //     return;
    // }

}

__device__ void comp_L(double* A, double* L, double* U, int i, int j, int n) {
    if (j >= i && j < n) {
    int l=0;
    double sum = 0.0;

    if (i==j) {
        L[j * n + j] = 1.0;
    }
    else {
        #pragma unroll
        for (l = 0; l < i; l++) {
            sum += L[j * n + l] * U[l * n + i];
        }
        L[j * n + i] = (A[j * n + i] - sum) / U[i * n + i];
    }
    }

    // else {
    //     return;
    // }
}

__global__ void lu_decomp(double* A, double* L, double* U, int n) {
    int i,j;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;
    // int res = n - blockSize;
    int ratio = n / blockSize;
    int rem = n % blockSize;
    int m = ratio - 1;

    // printf("rem: %d\n", rem);
    // printf("ratio: %d\n", ratio);

    for (i = 0; i < n; i++) {
                // printf("k: %d\n", i);

        if (j < n) {
            int k = 0;
            for (k = 0; k < ratio; k++) {
                comp_U(A, L, U, i, j + blockSize * k, n);
            }
        }
        // __syncthreads();
        // if (j < n) {
        if (rem >= 0 && rem < blockSize) {
            // int m = ratio - 1;
            // for loop with blockDim.x * ratio + j  and check if the new j index is within bounds of n
            if ((j + blockSize * m + rem) >= i && (j + blockSize * m + rem) < n){
                comp_U(A, L, U, i, j + blockSize * m + rem, n);
            }
        }
        // }

        // if (rem > 0 && rem < blockDim.x) {
        //     if ((j + res + blockDim.x + 1) >= i && (j + res + blockDim.x + 1) < n && (j + res + blockDim.x + 1) >= blockDim.x){
        //         comp_U(A, L, U, i, (j + res + blockDim.x + 1), n);
        //     }
        // }

        __syncthreads();

        if (j < n) {
            int k = 0;
            for (k = 0; k < ratio; k++) {
                comp_L(A, L, U, i, j + blockSize * k, n);
            }
        }
        // __syncthreads();
        // if (j < n) {
        if (rem >= 0 && rem < blockSize) {
            if ((j + blockSize * m + rem) >= i && (j + blockSize * m + rem) < n){
                comp_L(A, L, U, i, j + blockSize * m + rem, n);
            }
        }
        // }

        // if (rem > 0 && rem < blockDim.x) {
        //     if ((j + res + blockDim.x + 1) >= i && (j + res + blockDim.x + 1) < n && (j + res + blockDim.x + 1) >= blockDim.x){
        //         comp_L(A, L, U, i, (j + res + blockDim.x + 1), n);
        //     }
        // }

        __syncthreads();
    }
}

int main() {
    double *A, *L, *U, *LU;
    double *d_A, *d_L, *d_U;
    int n = 100;
    int i=0, j=0;
    int numElements = n * n;
    double startT=0.0, endT=0.0;

    A = (double *)malloc(numElements * sizeof(double));
    L = (double *)malloc(numElements * sizeof(double));
    U = (double *)malloc(numElements * sizeof(double));
    LU = (double *)malloc(numElements * sizeof(double));

    cudaMalloc(&d_A, numElements * sizeof(double));
    cudaMalloc(&d_L, numElements * sizeof(double));
    cudaMalloc(&d_U, numElements * sizeof(double));

    // double inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    FILE *file = fopen("matrix.txt", "r");

    srand(time(NULL));
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            // fscanf(file, "%lf", &A[i * n + j]);
            // A[i * n + j] = inputMatrix[i * n + j];
            A[i * n + j] = rand() % 10 + 1;
            L[i * n + j] = 0.0;
            U[i * n + j] = 0.0;
        }
    }

    cudaMemcpy(d_A, A, numElements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, numElements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, numElements * sizeof(double), cudaMemcpyHostToDevice);

    startT = clock();
    lu_decomp<<<1, 32>>>(d_A, d_L, d_U, n);
    endT = clock();

    cudaMemcpy(A, d_A, numElements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, numElements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, numElements * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Time taken: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    // // print A
    // printf("Orig A:\n");
    // for (i = 0; i < n; i++) {
    //     for (j = 0; j < n; j++) {
    //         printf("%f ", A[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // print U
    printf("U:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", U[i * n + j]);
        }
        printf("\n");
    }

    // // print L
    // printf("L:\n");
    // for (i = 0; i < n; i++) {
    //     for (j = 0; j < n; j++) {
    //         printf("%f ", L[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // multipy L and U to check if A = LU
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += L[i * n + k] * U[k * n + j];
                LU[i * n + j] = sum;
            }
            // printf("%f ", sum);
        }
        // printf("\n");
    }

    double diff=0.0;

    // subtract A from LU
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            diff += fabs(LU[i * n + j] - A[i * n + j]);
            // printf("%f ", fabs(LU[i * n + j] - A[i * n + j]));
        }
        // printf("\n");
    }

    if (fabs(diff) < 0.0001) {
        printf("LU decomposition is correct\n");
        printf("A = LU\n");
    } else {
        printf("LU decomposition is incorrect\n");
        printf("A != LU\n");
    }

    printf("Diff is %f\n", diff);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    free(A);
    free(L);
    free(U);
    free(LU);

    return 0;
}