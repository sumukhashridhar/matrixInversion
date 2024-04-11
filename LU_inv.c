#include "LU_inv.h"

void lu_decomposition(double* A, double* L, double* U, int N) {
    int i, j, k;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += L[i * N + k] * U[k * N + j];
            }
            U[i * N + j] = A[i * N + j] - sum;
        }

        for (j = i; j < N; j++) {
            if (i == j)
                L[i * N + i] = 1.0;
            else {
                sum = 0.0;
                for (k = 0; k < i; k++) {
                    sum += L[j * N + k] * U[k * N + i];
                }
                L[j * N + i] = (A[j * N + i] - sum) / U[i * N + i];
            }
        }
    }

    // print L
    printf("L:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", L[i * N + j]);
        }
        printf("\n");
    }

    // print U
    printf("U:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", U[i * N + j]);
        }
        printf("\n");
    }

}

void forward_substitution(double* L, double* b, double* y, int N) {
    int i, j;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        sum = 0.0;
        for (j = 0; j < i; j++) {
            sum += L[i * N + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * N + i];
    }
}

void backward_substitution(double* U, double* y, double* x, int N) {
    int i, j;
    double sum = 0.0;

    for (i = N-1; i >= 0; i--) {
        sum = 0.0;
        for (j = i+1; j < N; j++) {
            sum += U[i * N + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * N + i];
    }
}

void lu_invert_matrix(double* A, double* A_inv, int N) {
    int i, j;

    double* L = (double*)malloc(N * N * sizeof(double));
    double* U = (double*)malloc(N * N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double* e = (double*)malloc(N * sizeof(double));

    lu_decomposition(A, L, U, N);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j)
                e[j] = 1.0;
            else
                e[j] = 0.0;
        }
        forward_substitution(L, e, y, N);
        backward_substitution(U, y, &A_inv[i * N], N);
    }

    free(L);
    free(U);
    free(y);
    free(e);
}