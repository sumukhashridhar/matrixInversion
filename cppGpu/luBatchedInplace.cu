#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "luBatchedInplace.cuh"

int main() {
    int matrixSize, numMatrices, numThreads;

    // let user input matrix size and number of matrices
    std::cout << "Enter matrix size: ";
    std::cin >> matrixSize;
    std::cout << "Enter number of matrices: ";
    std::cin >> numMatrices;
    std::cout << "Enter number of threads: ";
    std::cin >> numThreads;
    
    int numElements = matrixSize * matrixSize * numMatrices;

    auto A = std::make_unique<std::vector<FpType>>(numElements);
    auto A_inv = std::make_unique<std::vector<FpType>>(numElements);

    FpType* d_A;
    cudaMalloc(&d_A, numElements * sizeof(FpType));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                (*A)[(i * matrixSize) + offset + j] = inputMatrix[(i * matrixSize) + j];
                (*A_inv)[(i * matrixSize) + offset + j] = 0.0;
            }
        }
    }

    cudaMemcpy(d_A, A->data(), numElements * sizeof(FpType), cudaMemcpyHostToDevice);

    int shMemSize = matrixSize * matrixSize * sizeof(FpType);

    cudaEventRecord(start, 0);
    batched_lu<<<numMatrices, numThreads, shMemSize>>>(d_A, matrixSize, numMatrices);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    cudaMemcpy(A_inv->data(), d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost);

    // print A_inv
    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        std::cout << "Matrix " << k + 1 << '\n';
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                std::cout << (*A_inv)[(i * matrixSize) + offset + j] << ' ';
            }
            std::cout << '\n';
        }
    }

    // auto startT = std::chrono::high_resolution_clock::now();
    // verifyInv(A->data(), A_inv->data(), matrixSize, numMatrices);
    // auto endT = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = endT - startT;
    // std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";
    
    cudaFree(d_A);

    return 0;
}
