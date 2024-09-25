#include "luShared.cuh"

int main() {
    FpType *A, *L, *U, *LU, *LandU;
    FpType *d_A;
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
    LandU = (FpType *)malloc(numElements * sizeof(FpType));

    cudaMalloc(&d_A, numElements * sizeof(FpType));

    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    FILE *file = fopen("matrix.txt", "r");

    srand(time(NULL));
    for (i = 0; i < matrixSize; i++) {
        for (j = 0; j < matrixSize; j++) {
            fscanf(file, "%lf", &A[i * matrixSize + j]);
            // A[i * matrixSize + j] = inputMatrix[i * matrixSize + j];
            // A[i * matrixSize + j] = rand() % 10 + 1;
            // L[i * matrixSize + j] = 0.0;
            // U[i * matrixSize + j] = 0.0;
            LandU[i * matrixSize + j] = 0.0;
        }
    }

    cudaMemcpy(d_A, A, numElements * sizeof(FpType), cudaMemcpyHostToDevice);

    int shMemSize = numElements * sizeof(FpType);

    startT = clock();
    lu_decomp<<<1, numThreads, shMemSize>>>(d_A, matrixSize);
    endT = clock();

    cudaMemcpy(LandU, d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);

    printf("Time taken: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    // // print A
    // printf("Orig A:\n");
    // for (i = 0; i < matrixSize; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", A[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }
    
    // // print LandU
    // printf("A from GPU:\n");
    // for (i = 0; i < matrixSize; i++) {
    //     for (j = 0; j < matrixSize; j++) {
    //         printf("%f ", LandU[i * matrixSize + j]);
    //     }
    //     printf("\n");
    // }



    // assign L and U
    for (i = 0; i < matrixSize; i++) {
        for (j = 0; j < matrixSize; j++) {
            if (i > j) {
                L[i * matrixSize + j] = LandU[i * matrixSize + j];
                U[i * matrixSize + j] = 0.0;
            }
            else if (i == j) {
                L[i * matrixSize + j] = 1.0;
                U[i * matrixSize + j] = LandU[i * matrixSize + j];
            }
            else {
                L[i * matrixSize + j] = 0.0;
                U[i * matrixSize + j] = LandU[i * matrixSize + j];
            }
        }
    }

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
                LU[i * matrixSize + j] = sum;
            }
        }
    }

    FpType diff=0.0;

    // subtract A from LU
    for (i = 0; i < matrixSize; i++) {
        for (j = 0; j < matrixSize; j++) {
            diff += fabs(LU[i * matrixSize + j] - A[i * matrixSize + j]);
        }
    }

    if (fabs(diff) < 10e-6) {
        printf("LU decomposition is correct\n");
    }
    if (fabs(diff) > 10e-6) {
        printf("LU decomposition is incorrect\n");
    }

    printf("Diff is %f\n", diff);

    cudaFree(d_A);

    free(A);
    free(L);
    free(U);
    free(LU);
    free(LandU);

    return 0;
}
