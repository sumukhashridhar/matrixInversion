#include "LU_inv.h"

void lu_decomp(int N) {
    int i, j, k;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += data.L[i * N + k] * data.U[k * N + j]; // matrix matrix mult
            }
            data.U[i * N + j] = data.A[i * N + j] - sum;
        }

        for (j = i; j < N; j++) {
            if (i == j)
                data.L[i * N + i] = 1.0; // set diagonal as 1.0 like in Doolittle's algorithm
            else {
                sum = 0.0;
                for (k = 0; k < i; k++) {
                    sum += data.L[j * N + k] * data.U[k * N + i]; // matrix matrix mult
                }
                data.L[j * N + i] = (data.A[j * N + i] - sum) / data.U[i * N + i];
            }
        }
    }

    // print L
    printf("L:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", data.L[i * N + j]);
        }
        printf("\n");
    }

    // print U
    printf("U:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", data.U[i * N + j]);
        }
        printf("\n");
    }

}

void forward_sub(double* b, int N) {
    int i, j;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        sum = 0.0;
        for (j = 0; j < i; j++) {
            sum += data.L[i * N + j] * data.y[j]; // matrix vector mult
        }
        data.y[i] = (b[i] - sum) / data.L[i * N + i];
    }
}

void backward_sub(double* x, int N) {
    int i, j;
    double sum = 0.0;

    for (i = N-1; i >= 0; i--) {
        sum = 0.0;
        for (j = i+1; j < N; j++) {
            sum += data.U[i * N + j] * x[j]; // matrix vector mult
        }
        x[i] = (data.y[i] - sum) / data.U[i * N + i];
    }
}

void lu_invert_matrix(int N) {
    int i, j;
    int numElems = N * N;

    lu_decomp(N);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j)
                data.e[j] = 1.0;
            else
                data.e[j] = 0.0;
        }
        forward_sub(data.e, N);
        backward_sub(&data.A_inv[i * N], N);
    }

}
