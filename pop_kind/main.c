#include "LU_inv.h"

int main() {
    int i, j, k;
    int N = 3;
    int numElems = N * N;
    
    double result = 0.0;
    double startT=0.0, endT=0.0;

    // assign memory for the input and inverse matrix
    double *A = (double*)malloc(N * N * sizeof(double));
    double *A_inv = (double*)malloc(N * N * sizeof(double));
    double *Id = (double*)malloc(N * N * sizeof(double));

    // init the input and inverse matrix
    double inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};
    
    // populate the input and inverse matrix
    for (i = 0; i < numElems; ++i) {
        A[i] = inputMatrix[i];
        A_inv[i] = 0.0;
        Id[i] = 0.0;
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
                Id[j * N + i] = result;
            }
        printf("%f ", result);
        }
    printf("\n");
    }

    // check if the multiplication is correct
    int idCnt = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (Id[j * N + i] == 1.0) {
                idCnt++;
            }
        }
    }

    if (idCnt == N) {
        printf("Inverse is correct\n");
    } else {
        printf("Inverse is incorrect\n");
    }

    // free the memory
    free(A);
    free(A_inv);

    return 0;
}
