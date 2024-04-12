#include "LU_inv.h"

void assign_memory(int N) {
    data.A = (double*)malloc(N * N * sizeof(double));
    data.L = (double*)malloc(N * N * sizeof(double));
    data.U = (double*)malloc(N * N * sizeof(double));
    data.y = (double*)malloc(N * sizeof(double));
    data.e = (double*)malloc(N * sizeof(double));
    data.A_inv = (double*)malloc(N * N * sizeof(double));
    data.Id = (double*)malloc(N * N * sizeof(double));

}

void fill_matrices(int N) {
    int i, j;
    int numElems = N * N;
    double inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    for (i = 0; i < numElems; ++i) {
        data.A[i] = inputMatrix[i];
        data.A_inv[i] = 0.0;
        data.L[i] = 0.0;
        data.U[i] = 0.0;
        data.Id[i] = 0.0;
    }

    for (i = 0; i < N; i++) {
        data.y[i] = 0.0;
    }
}

void free_memory() {
    free(data.A);
    free(data.L);
    free(data.U);
    free(data.y);
    free(data.e);
    free(data.A_inv);
}