#include "luBatchedInplace.cuh"

int main() {
    int matrixSize, numMatrices, numThreads;

    std::cout << "Enter matrix size: ";
    std::cin >> matrixSize;
    std::cout << "Enter number of matrices: ";
    std::cin >> numMatrices;
    std::cout << "Enter number of threads: ";
    std::cin >> numThreads;
    // numThreads = 30;
    
    int numElements = matrixSize * matrixSize * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    FpType* d_A;
    cudaMalloc(&d_A, numElements * sizeof(FpType));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        std::ifstream file("matrix.txt");
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                // file >> A[(i * matrixSize) + offset + j];
                A[(i * matrixSize) + offset + j] = inputMatrix[(i * matrixSize) + j];
                // A[(i * matrixSize) + offset + j] = rand() % 10;
                A_inv[(i * matrixSize) + offset + j] = 0.0;
            }
        }
        file.close();
    }

    cudaMemcpy(d_A, A.data(), numElements * sizeof(FpType), cudaMemcpyHostToDevice);
    std::cout << "Data copied to device." << '\n';

    int shMemSize = matrixSize * matrixSize * sizeof(FpType);
    cudaFuncSetAttribute(batched_lu, cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize);

    cudaEventRecord(start, 0);
    batched_lu<<<numMatrices, numThreads, shMemSize>>>(d_A, matrixSize, numMatrices);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    cudaMemcpy(A_inv.data(), d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);
    std::cout << "Data copied back to host." << '\n';

    // // print A_inv
    // for (int k = 0; k < numMatrices; ++k) {
    //     int offset = k * matrixSize * matrixSize;
    //     std::cout << "Matrix " << k << ":\n";
    //     for (int i = 0; i < matrixSize; ++i) {
    //         for (int j = 0; j < matrixSize; ++j) {
    //             std::cout << A_inv[(i * matrixSize) + offset + j] << " ";
    //         }
    //         std::cout << '\n';
    //     }
    // }

    auto startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";
    
    cudaFree(d_A);

    return 0;
}
