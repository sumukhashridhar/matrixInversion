#include "luBatched.cuh"

int main() {
    int matrixSize, numMatrices, numThreads;

    // let user input matrix size and number of matrices
    printf("Enter matrix size: ");
    scanf("%d", &matrixSize);
    printf("Enter number of matrices: ");
    scanf("%d", &numMatrices);
    printf("Enter number of threads: ");
    scanf("%d", &numThreads);
    
    FpType *A, *L, *U, *LU, *LandU;
    FpType *d_A, *d_L, *d_U;
    int i=0, j=0;
    int numElements = matrixSize * matrixSize * numMatrices;
    FpType startT=0.0, endT=0.0;

    A = (FpType *)malloc(numElements * sizeof(FpType));
    L = (FpType *)malloc(numElements * sizeof(FpType));
    U = (FpType *)malloc(numElements * sizeof(FpType));
    LU = (FpType *)malloc(matrixSize * matrixSize * sizeof(FpType));
    LandU = (FpType *)malloc(numElements * sizeof(FpType));

    cudaMalloc(&d_A, numElements * sizeof(FpType));
    cudaMalloc(&d_L, numElements * sizeof(FpType));
    cudaMalloc(&d_U, numElements * sizeof(FpType));

    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    FILE *f; // = fopen("matrix.txt", "r");

    for (int k = 0; k < numMatrices; k++) {
        int offset = k * matrixSize * numMatrices;
        f = fopen("matrix.txt", "r");
        for (i = 0; i < matrixSize; i++) {
            for (j = 0; j < matrixSize; j++) {
                // fscanf(f, "%lf", &A[(i * matrixSize) + offset + j]);
                A[(i * matrixSize) + offset + j] = inputMatrix[i * matrixSize + j];
                // A[(i * matrixSize) + offset + j] = rand() % 10 + 1;
                L[(i * matrixSize) + offset + j] = 0.0;
                U[(i * matrixSize) + offset + j] = 0.0;
                LandU[(i * matrixSize) + offset + j] = 0.0;
            }
        }
    }

    cudaMemcpy(d_A, A, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, numElements * sizeof(FpType), cudaMemcpyHostToDevice);

    int shMemSize = matrixSize * matrixSize * sizeof(FpType);

    startT = clock();
    batched_lu<<<numMatrices, numThreads>>>(d_A, d_L, d_U, matrixSize, numMatrices);
    endT = clock();

    cudaMemcpy(A, d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);

    // // assign L and U
    // for (int k = 0; k < numMatrices; k++) {
    //     int offset = k * matrixSize * numMatrices;
    //     for (i = 0; i < matrixSize; i++) {
    //         for (j = 0; j < matrixSize; j++) {
    //             if (i > j) {
    //                 L[(i * matrixSize) + offset + j] = LandU[(i * matrixSize) + offset + j];
    //                 U[(i * matrixSize) + offset + j] = 0.0;
    //             }
    //             else if (i == j) {
    //                 L[(i * matrixSize) + offset + j] = 1.0;
    //                 U[(i * matrixSize) + offset + j] = LandU[(i * matrixSize) + offset + j];
    //             }
    //             else {
    //                 L[(i * matrixSize) + offset + j] = 0.0;
    //                 U[(i * matrixSize) + offset + j] = LandU[(i * matrixSize) + offset + j];
    //             }
    //         }
    //     }
    // }

    // // print all L and U
    // for (int k = 0; k < numMatrices; k++) {
    //     int offset = k * matrixSize * numMatrices;
    //     printf("Matrix %d\n", k+1);
    //     printf("L\n");
    //     for (i = 0; i < matrixSize; i++) {
    //         for (j = 0; j < matrixSize; j++) {
    //             printf("%f ", L[(i * matrixSize) + offset + j]);
    //         }
    //         printf("\n");
    //     }

    //     printf("U\n");
    //     for (i = 0; i < matrixSize; i++) {
    //         for (j = 0; j < matrixSize; j++) {
    //             printf("%f ", U[(i * matrixSize) + offset + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // int k=0, l=0, offset=0, wrngLU=0, corrLU=0, nanLU=0;
    // FpType sum=0.0, diff=0.0;

    // for (k = 0; k < numMatrices; k++) {
    //     for (i = 0; i < matrixSize; i++) {
    //         for (j = 0; j < matrixSize; j++) {
    //             sum = 0.0;
    //             offset = k * numMatrices * matrixSize;
    //             for (l = 0; l < matrixSize; l++) {
    //                 sum += L[(i * matrixSize) + offset + l] * U[(l * matrixSize) + offset + j];
    //                 LU[(i * matrixSize) + j] = sum;
    //             }
    //         }
    //     }

    //     // subtract A from LU
    //     diff = 0.0;
    //     for (i = 0; i < matrixSize; i++) {
    //         for (j = 0; j < matrixSize; j++) {
    //             diff += fabs(LU[i * matrixSize + j] - A[(i * matrixSize) + offset + j]);
    //         }
    //     }

    //     if (isnan(diff))  {
    //         nanLU++;
    //         printf("Diff is nan\n");
    //     }

    //     if (fabs(diff) < 10e-6) {
    //         corrLU++;
    //         printf("Correct diff is %f\n", diff);
    //         // printf("LU decomposition is correct\n");
    //         // printf("A = LU\n");
    //     }
        
    //     if (fabs(diff) > 10e-6) {
    //         wrngLU++;
    //         printf("Incorrect diff is %f\n", diff);
    //         // printf("LU decomposition is incorrect\n");
    //         // printf("A != LU\n");
    //     }

    //     // printf("Diff is %f\n", diff);
    // }

    // printf("Matrix size: %d\n", matrixSize);
    // printf("Number of matrices: %d\n", numMatrices);
    // printf("Number of correct LU decompositions: %d\n", corrLU);
    // printf("Number of incorrect LU decompositions: %d\n", wrngLU);
    // printf("Number of nan LU decompositions: %d\n", nanLU);

    printf("Time taken: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    // print A
    printf("A\n");
    for (int i = 0; i < numMatrices * matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%f ", A[i * matrixSize + j]);
        }
        printf("\n");
    }

    // print U
    printf("U\n");
    for (int i = 0; i < numMatrices * matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%f ", U[i * matrixSize + j]);
        }
        printf("\n");
    }

    // print L
    printf("L\n");
    for (int i = 0; i < numMatrices * matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%f ", L[i * matrixSize + j]);
        }
        printf("\n");
    }

    // save L and U to file
    FILE *fL = fopen("L.txt", "w");
    FILE *fU = fopen("U.txt", "w");

    for (int i = 0; i < numMatrices * matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            fprintf(fL, "%f ", L[i * matrixSize + j]);
            fprintf(fU, "%f ", U[i * matrixSize + j]);
        }
        fprintf(fL, "\n");
        fprintf(fU, "\n");
    }

    fclose(fL);
    fclose(fU);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    free(A);
    free(L);
    free(U);
    free(LU);
    free(LandU);

    return 0;
}
