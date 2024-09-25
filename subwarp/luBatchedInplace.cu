#include "luBatchedInplace.cuh"

int main() {
    int matrixSize, numMatrices, numThreads;

    std::cout << "Enter matrix size: ";
    std::cin >> matrixSize;
    std::cout << "Enter number of matrices: ";
    std::cin >> numMatrices;
    std::cout << "Enter number of threads in a block: ";
    std::cin >> numThreads;
    // numThreads = 32;
    int threadsPerMatrix = matrixSize;
    int matricesPerBlock = numThreads / threadsPerMatrix;
    int numBlocks = numMatrices / matricesPerBlock;

    int numElements = matrixSize * matrixSize * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    FpType* d_A;
    CUDA_CHECK(cudaMallocManaged(&d_A, numElements * sizeof(FpType)));

    FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};

    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        std::ifstream file("matrix.txt");
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                file >> A[(i * matrixSize) + offset + j];
                // A[(i * matrixSize) + offset + j] = inputMatrix[(i * matrixSize) + j];
                // A[(i * matrixSize) + offset + j] = rand() % 10;
                A_inv[(i * matrixSize) + offset + j] = static_cast<FpType>(0.0);
            }
        }
        file.close();
    }

    // printMatrices(A, matrixSize, numMatrices);

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), numElements * sizeof(FpType), cudaMemcpyHostToDevice));
    std::cout << "Data copied to device." << '\n';

    int shMemSize = matricesPerBlock * matrixSize * matrixSize * sizeof(FpType);
    CUDA_CHECK(cudaFuncSetAttribute(batched_lu_subwarp<FpType>, cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    batched_lu_subwarp<FpType><<<numBlocks, numThreads, shMemSize>>>(d_A, matrixSize, numMatrices, threadsPerMatrix, matricesPerBlock);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    CUDA_CHECK(cudaMemcpy(A_inv.data(), d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost));
    std::cout << "Data copied back to host." << '\n';

    // print A_inv
    // printMatrices(A_inv, matrixSize, numMatrices);

    auto startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";
    
    cudaFree(d_A);

    return 0;
}
