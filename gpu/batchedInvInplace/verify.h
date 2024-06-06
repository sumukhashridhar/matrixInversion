#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double FpType;

void verifyLU(FpType *A, FpType *L, FpType *U, int matrixSize, int numMatrices) {
    int offset = 0, wrngLU = 0, corrLU = 0, nanLU = 0;
    FpType sum = 0.0, diff = 0.0;

    #pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; k++) {
        offset = k * matrixSize * matrixSize;
        diff = 0.0;
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                sum = 0.0;
                #pragma omp simd reduction(+ : sum)
                for (int l = 0; l < matrixSize; l++) {
                    sum += L[(i * matrixSize) + offset + l] * U[(l * matrixSize) + offset + j];
                }
                diff = fabs(sum - A[(i * matrixSize) + offset + j]);
            }
        }

        if (isnan(diff)) {
            nanLU++;
        }

        if (fabs(diff) < 10e-6) {
            corrLU++;
        }

        if (fabs(diff) > 10e-6) {
            wrngLU++;
        }
    }

    printf("Correct LU decompositions: %d\n", corrLU);
    printf("Incorrect LU decompositions: %d\n", wrngLU);
    printf("NaN LU decompositions: %d\n", nanLU);
}

void verifyInv(FpType *A, FpType *A_inv, int matrixSize, int numMatrices) {
    FpType result = 0.0;
    int offset = 0, idCnt = 0, corrInv = 0, wrngInv = 0;

    // Multiplication of A and A_inv
    #pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; k++) {
        offset = k * matrixSize * matrixSize;
        idCnt = 0;
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                result = 0.0;
                #pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; l++) {
                    result += A[(i * matrixSize) + offset + l] * A_inv[(j * matrixSize) + offset + l];
                }
                // printf("result: %f\n", result);
                if (i == j) {
                    if (fabs(result - 1.0) < 10e-6) {
                        idCnt++;
                    }
                }
            }
            // printf("\n");
        }

        if (idCnt == matrixSize) {
            corrInv++;
        }
        else {
            wrngInv++;
        }
    }

    printf("Correct inversions: %d\n", corrInv);
    printf("Incorrect inversions: %d\n", wrngInv);
}
