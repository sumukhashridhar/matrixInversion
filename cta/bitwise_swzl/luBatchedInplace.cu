#include "luBatchedInplace.cuh"

int main() {
    constexpr int matrixSize=MATRIXSIZE;
    constexpr int numMatrices=NUMMATRICES;
    constexpr int numThreads=NUMTHREADS;

    const int threadsPerMatrix = matrixSize;
    const int matricesPerBlock = numThreads / threadsPerMatrix;
    const int numBlocks = numMatrices / matricesPerBlock;

    std::cout << "Matrix size: " << matrixSize << '\n';
    std::cout << "Number of matrices: " << numMatrices << '\n'; 
    std::cout << "Number of threads per block: " << numThreads << '\n';
    std::cout << "Threads per matrix: " << threadsPerMatrix << '\n';
    std::cout << "Matrices per block: " << matricesPerBlock << '\n';
    std::cout << "Number of blocks: " << numBlocks << '\n';

    int numElements = matrixSize * matrixSize * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    FpType* d_A;
    CUDA_CHECK(cudaMallocManaged(&d_A, numElements * sizeof(FpType)));

    // FpType inputMatrix[] = {2, 3, 4, 5};//, 10, 4, 2, 4, 2};
    //  FpType inputMatrix[] = {4, 11, 3, 4};//, 10, 4, 2, 4, 2};
    // FpType inputMatrix[] = {2, 7, 1, 5, 3, -2, 0, 1, 1, 5, 3, 4, 7, 3, 2, 8};
    // make input matrix as Identity matrix
    // FpType inputMatrix[] = {10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10};

    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        std::ifstream file("matrix.txt");
        // std::ifstream file("main_mtrx.txt");
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
    CUDA_CHECK(cudaFuncSetAttribute(batched_lu_subwarp<FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices>, cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    batched_lu_subwarp<FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices><<<numBlocks, numThreads, shMemSize>>>(d_A);//, numMatrices);//, matricesPerBlock);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    CUDA_CHECK(cudaMemcpy(A_inv.data(), d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost));
    std::cout << "Data copied back to host." << '\n';

    // print A_inv
    // printf("\n");
    // printMatrices(A_inv, matrixSize, numMatrices);
    // printf("\n");
    // printMatrices(A, matrixSize, numMatrices);

    // write A_inv to file
    // auto filename = "8thread32_mtrx.txt";
    // writeToFile(A_inv, filename, matrixSize, numMatrices);

    auto startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    // verifyLU(A, A_inv, matrixSize, numMatrices);
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";

    cudaFree(d_A);

    return 0;
}
