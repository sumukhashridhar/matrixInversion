#include "LU_inv.h"

int main() {
    int i, j, k;
    int N = 3;

    double result = 0.0;
    double startT=0.0, endT=0.0;

    // assign memory for the memory allocation struct
    assign_memory(N);

    // fill the matrices
    fill_matrices(N);

    startT = clock();
    lu_invert_matrix(N);
    endT = clock();

    // time taken in seconds
    printf("Time taken: %fs\n", (endT - startT) / CLOCKS_PER_SEC);

    printf("Inverted Matrix:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", data.A_inv[j * N + i]);
        }
        printf("\n");
    }

    printf("Multiplication of A and A_inv:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            result = 0.0;
            for (k = 0; k < N; k++) {
                result += data.A[i * N + k] * data.A_inv[j * N + k];
                data.Id[j * N + i] = result;
            }
        printf("%f ", result);
        }
    printf("\n");
    }

    // check if the multiplication is correct
    int idCnt = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (data.Id[j * N + i] == 1.0) {
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
    free_memory();

    return 0;
}
