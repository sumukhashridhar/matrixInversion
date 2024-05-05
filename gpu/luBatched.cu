#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void batched_lu(float* A, float* L, float* U, int n, int numMatrices) {
    int b=0;
    b = blockIdx.x;

    if (b >= 0 && b < numMatrices) {
        int j = threadIdx.x;
        int i = 0, k = 0;
        int matrixSize = n;

        for (i = 0; i < matrixSize; i++) {
            if (j >= i && j < matrixSize) {

                float sum = 0.0;
                for (k = 0; k < i; k++) {
                    sum += L[(i * matrixSize) + (b * numMatrices * matrixSize) + k] * U[(k * matrixSize) + (b * numMatrices * matrixSize) + j];
                }
                U[(i * matrixSize) + (b * numMatrices * matrixSize) + j] = A[(i * matrixSize) + (b * numMatrices * matrixSize) + j] - sum;
            }

            __syncthreads();

            if (j >= i && j < matrixSize) {
                if (i==j) {
                    L[(j * matrixSize) + (b * numMatrices * matrixSize) + j] = 1.0;
                }
                else {
                    float sum = 0.0;
                    for (k = 0; k < i; k++) {
                        sum += L[(j * matrixSize) + (b * numMatrices * matrixSize) + k] * U[(k * matrixSize) + (b * numMatrices * matrixSize) + i];
                    }
                    L[(j * matrixSize) + (b * numMatrices * matrixSize) + i] = (A[(j * matrixSize) + (b * numMatrices * matrixSize) + i] - sum) / U[(i * matrixSize) + (b * numMatrices * matrixSize) + i];
                }
            }
            __syncthreads();
        }
    }
}

int main() {
    int matrixSize=0, numMatrices =0;

    // let user input matrix size and number of matrices
    printf("Enter matrix size: ");
    scanf("%d", &matrixSize);
    printf("Enter number of matrices: ");
    scanf("%d", &numMatrices);
    
    float *A, *L, *U, *LU;
    float *d_A, *d_L, *d_U;
    int i=0, j=0;
    int numElements = matrixSize * matrixSize * numMatrices;
    float startT=0.0, endT=0.0;

    A = (float *)malloc(numElements * sizeof(float));
    L = (float *)malloc(numElements * sizeof(float));
    U = (float *)malloc(numElements * sizeof(float));
    LU = (float *)malloc(matrixSize * matrixSize * sizeof(float));

    cudaMalloc(&d_A, numElements * sizeof(float));
    cudaMalloc(&d_L, numElements * sizeof(float));
    cudaMalloc(&d_U, numElements * sizeof(float));

    // float matr[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    // float inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                        4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                        4, 11, 3, 4, 10, 4, 2, 4, 2
                        //    4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                        4, 11, 3, 4, 10, 4, 2, 4, 2
    //                     //    4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                     //    4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                     //    4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                     //    4, 11, 3, 4, 10, 4, 2, 4, 2,
    //                     //    4, 11, 3, 4, 10, 4, 2, 4, 2
                        //    };

    srand(time(NULL));
    for (i = 0; i < matrixSize * numMatrices; i++) {
        for (j = 0; j < matrixSize; j++) {
            // A[i * matrixSize + j] = inputMatrix[i * matrixSize + j];
            A[i * matrixSize + j] = rand() % 10 + 1;
            L[i * matrixSize + j] = 0.0;
            U[i * matrixSize + j] = 0.0;
        }
    }

    cudaMemcpy(d_A, A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, numElements * sizeof(float), cudaMemcpyHostToDevice);

    startT = clock();
    batched_lu<<<numMatrices, matrixSize>>>(d_A, d_L, d_U, matrixSize, numMatrices);
    cudaDeviceSynchronize();
    endT = clock();

    cudaMemcpy(A, d_A, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Time taken: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    // // print A
    // printf("Orig A:\n");
    // for (i = 0; i < numMatrices * matrixSize; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", A[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }

    // // print U
    // printf("U:\n");
    // for (i = 0; i < matrixSize * numMatrices; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", U[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }

    // // print L
    // printf("L:\n");
    // for (i = 0; i < matrixSize * numMatrices; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", L[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }

    int k=0, l=0, wrngLu=0;
    float sum=0.0, diff=0.0;

    for (k = 0; k < numMatrices; k++) {
        diff = 0.0;
        for (i = 0; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                sum = 0.0;
                for (l = 0; l < matrixSize; l++) {
                    sum += L[(i * matrixSize) + (k * numMatrices * matrixSize) + l] * U[(l * matrixSize) + (k * numMatrices * matrixSize) + j];
                    LU[(i * matrixSize) + j] = sum;
                }
            }
        }
        // subtract A from LU
        for (i = 0; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                diff += fabs(LU[i * matrixSize + j] - A[(i * matrixSize) + (k * numMatrices * matrixSize) + j]);
            }
        }

        // if (fabs(diff) < 10e-6) {
        //     printf("LU decomposition is correct\n");
        //     printf("A = LU\n");
        // }
        
        if (fabs(diff) > 10e-6) {
            wrngLu++;
            // printf("LU decomposition is incorrect\n");
            // printf("A != LU\n");
        }

        // printf("Diff is %f\n", diff);
    }

    printf("Matrix size: %d\n", matrixSize);
    printf("Number of matrices: %d\n", numMatrices);
    printf("Number of incorrect LU decompositions: %d\n", wrngLu);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    free(A);
    free(L);
    free(U);
    free(LU);

    return 0;
}