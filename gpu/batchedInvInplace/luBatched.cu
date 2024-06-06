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
    
    FpType *A, *A_inv;
    FpType *d_A;
    int numElements = matrixSize * matrixSize * numMatrices;
    FpType startT=0.0, endT=0.0;

    A = (FpType *)malloc(numElements * sizeof(FpType));
    A_inv = (FpType *)malloc(numElements * sizeof(FpType));

    cudaMalloc(&d_A, numElements * sizeof(FpType));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // FpType inputMatrix[] = {4.3, 11.4, 3.1, 4.9, 10.2, 4.5, 2.6, 4.7, 2.1};
    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    for (int k = 0; k < numMatrices; k++) {
        int offset = k * matrixSize * matrixSize;
        // FILE *fl = fopen("matrix.txt", "r");
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                // fscanf(fl, "%lf", &A[(i * matrixSize) + offset + j]);
                A[(i * matrixSize) + offset + j] = inputMatrix[(i * matrixSize) + j];
                A_inv[(i * matrixSize) + offset + j] = 0.0;
            }
        }
        // fclose(fl);
    }

    cudaMemcpy(d_A, A, numElements * sizeof(FpType), cudaMemcpyHostToDevice);

    int shMemSize = matrixSize * matrixSize * sizeof(FpType);

    cudaEventRecord(start, 0);
    batched_lu<<<numMatrices, numThreads, shMemSize>>>(d_A, matrixSize, numMatrices);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaMemcpy(A_inv, d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);

    // print A_inv
    for (int k = 0; k < numMatrices; k++) {
        int offset = k * matrixSize * matrixSize;
        printf("Matrix %d\n", k + 1);
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                printf("%f ", A_inv[(i * matrixSize) + offset + j]);
            }
            printf("\n");
        }
    }

    // startT = clock();
    // verifyLU(A, L, U, matrixSize, numMatrices);
    // endT = clock();
    // printf("Time taken to verify LU: %f\n", (endT - startT) / CLOCKS_PER_SEC);

    startT = clock();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    endT = clock();
    printf("Time taken to verify inverse: %f\n", (endT - startT) / CLOCKS_PER_SEC);
    
    cudaFree(d_A);

    free(A);
    free(A_inv);

    return 0;
}
