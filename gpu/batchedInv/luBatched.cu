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
    
    FpType *A, *L, *U, *e, *y, *A_inv;
    FpType *d_A, *d_L, *d_U, *d_e, *d_y, *d_A_inv;
    int numElements = matrixSize * matrixSize * numMatrices;
    FpType startT=0.0, endT=0.0;

    A = (FpType *)malloc(numElements * sizeof(FpType));
    L = (FpType *)malloc(numElements * sizeof(FpType));
    U = (FpType *)malloc(numElements * sizeof(FpType));
    e = (FpType *)malloc(numElements * sizeof(FpType));
    y = (FpType *)malloc(numElements * sizeof(FpType));
    A_inv = (FpType *)malloc(numElements * sizeof(FpType));

    cudaMalloc(&d_A, numElements * sizeof(FpType));
    cudaMalloc(&d_L, numElements * sizeof(FpType));
    cudaMalloc(&d_U, numElements * sizeof(FpType));
    cudaMalloc(&d_e, numElements * sizeof(FpType));
    cudaMalloc(&d_y, numElements * sizeof(FpType));
    cudaMalloc(&d_A_inv, numElements * sizeof(FpType));

    FpType inputMatrix[] = {4.3, 11.4, 3.1, 4.9, 10.2, 4.5, 2.6, 4.7, 2.1};

    // FILE *f; // = fopen("matrix.txt", "r");

    for (int k = 0; k < numMatrices; k++) {
        int offset = k * matrixSize * matrixSize;
        // FILE *f = fopen("matrix.txt", "r");
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                // fscanf(f, "%lf", &A[(i * matrixSize) + offset + j]);
                // A[(i * matrixSize) + offset + j] = inputMatrix[(i * matrixSize) + j];
                A[(i * matrixSize) + offset + j] = rand() % 10 + 1;
                L[(i * matrixSize) + offset + j] = 0.0;
                U[(i * matrixSize) + offset + j] = 0.0;
                if (i == j) {
                    e[(i * matrixSize) + offset + j] = 1.0;
                }
                else {
                    e[(i * matrixSize) + offset + j] = 0.0;
                }
                y[(i * matrixSize) + offset + j] = 0.0;
                A_inv[(i * matrixSize) + offset + j] = 0.0;
            }
        }
        // fclose(f);
    }

    cudaMemcpy(d_A, A, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_inv, A_inv, numElements * sizeof(FpType), cudaMemcpyHostToDevice);

    int shMemSize = matrixSize * matrixSize * sizeof(FpType);

    startT = clock();
    batched_lu<<<numMatrices, numThreads>>>(d_A, d_L, d_U, d_e, d_y, d_A_inv, matrixSize, numMatrices);
    endT = clock();

    printf("Time taken: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    cudaMemcpy(A, d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, d_L, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    cudaMemcpy(A_inv, d_A_inv, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);

    verifyLU(A, L, U, matrixSize, numMatrices);
    verifyInv(A, A_inv, matrixSize, numMatrices);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_e);
    cudaFree(d_y);
    cudaFree(d_A_inv);

    free(A);
    free(L);
    free(U);
    free(e);
    free(y);
    free(A_inv);

    return 0;
}
