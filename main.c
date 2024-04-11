#include "LU_inv.h"

int main() {
    int i, j, k;
    int N = 3;
    int numElems = N * N;
    
    double result = 0.0;
    double startT=0.0, endT=0.0;

    // assign memory for the input and inverse matrix
    double* A = (double*)malloc(N * N * sizeof(double));
    double* A_inv = (double*)malloc(N * N * sizeof(double));

    // init the input and inverse matrix
    double inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};
    double initA_inv[] = {0, 0, 0,0, 0, 0,0, 0, 0};

    // populate the input and inverse matrix
    for (i = 0; i < numElems; ++i) {
        A[i] = inputMatrix[i];
        A_inv[i] = initA_inv[i];
    }

    startT = clock();
    lu_invert_matrix(A, A_inv, N);
    endT = clock();

    // time taken in seconds
    printf("Time taken: %fs\n", (endT - startT) / CLOCKS_PER_SEC);

    printf("Inverted Matrix:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", A_inv[j * N + i]);
        }
        printf("\n");
    }

    printf("Multiplication of A and A_inv:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            result = 0.0;
            for (k = 0; k < N; k++) {
                result += A[i * N + k] * A_inv[j * N + k];
            }
        printf("%f ", result);
        }
    printf("\n");
    }

    // free the memory
    free(A);
    free(A_inv);

    return 0;
}
