#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void lu_decomp(double* A, double* L, double* U, int n) {
    int i, j, k;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    for (i = 0; i < n; i++) {

        if (j >= i && j < n) {
            double sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += L[i * n + k] * U[k * n + j];
            }
            U[i * n + j] = A[i * n + j] - sum;
        }

        __syncthreads();

        if (j >= i && j < n) {
            if (i==j) {
                L[j * n + j] = 1.0;
            }
            else {
                double sum = 0.0;
                for (k = 0; k < i; k++) {
                    sum += L[j * n + k] * U[k * n + i];
                }
                L[j * n + i] = (A[j * n + i] - sum) / U[i * n + i];
            }
        }
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

    double inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

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
    lu_decomp<<<1, 128>>>(d_A, d_L, d_U, n);
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

    // // print U
    // printf("U:\n");
    // for (i = 0; i < n; i++) {
    //     for (j = 0; j < n; j++) {
    //         printf("%f ", U[i * n + j]);
    //     }
    //     printf("\n");
    // }

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