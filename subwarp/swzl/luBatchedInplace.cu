#include "luBatchedInplace.cuh"

int main() {
    constexpr int matrixSize=MATRIXSIZE;
    constexpr int numMatrices=NUMMATRICES;
    constexpr int numThreads=NUMTHREADS;

    const int threadsPerMatrix = matrixSize;
    // const int threadsPerMatrix = 32;
    const int matricesPerBlock = numThreads / threadsPerMatrix;
    const int numBlocks = numMatrices / matricesPerBlock;

    std::cout << "Matrix size: " << matrixSize << '\n';
    std::cout << "Number of matrices: " << numMatrices << '\n'; 
    std::cout << "Number of threads per block: " << numThreads << '\n';
    std::cout << "Threads per matrix: " << threadsPerMatrix << '\n';
    std::cout << "Matrices per block: " << matricesPerBlock << '\n';
    std::cout << "Number of blocks: " << numBlocks << '\n';

    // FpType inputMatrix[] = {2, 3, 4, 5};//, 10, 4, 2, 4, 2};
    //  FpType inputMatrix[] = {4, 11, 3, 4, 10, 4, 2, 4, 2};
    // FpType inputMatrix[] = {2, 7, 1, 5, 3, -2, 0, 1, 1, 5, 3, 4, 7, 3, 2, 8};
    // make input matrix as Identity matrix
    // FpType inputMatrix[] = {10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10};

    int numElements = matrixSize * matrixSize * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    std::cout << "Reading data from file." << '\n';

    // Read the file once into a single matrix-sized buffer
    std::ifstream file("matrix.txt");
    std::vector<FpType> templateMatrix(matrixSize * matrixSize);
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        file >> templateMatrix[i];
    }
    file.close();

    auto startT = std::chrono::high_resolution_clock::now();
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;

    startT = std::chrono::high_resolution_clock::now();
    #pragma acc parallel loop
    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        
        #pragma acc loop vector
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                int templateIndex = (i * matrixSize) + j;
                A[(i * matrixSize) + offset + j] = templateMatrix[templateIndex];
                A_inv[(i * matrixSize) + offset + j] = static_cast<FpType>(0.0);
            }
        }
    }
    endT = std::chrono::high_resolution_clock::now();
    elapsed = endT - startT;
    std::cout << "Time taken to read data: " << elapsed.count() << " seconds\n";
    std::cout << "Data read from file." << '\n';

    FpType* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, numElements * sizeof(FpType)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), numElements * sizeof(FpType), cudaMemcpyHostToDevice));
    std::cout << "Data copied to device." << '\n';

    int shMemSize = matricesPerBlock * matrixSize * matrixSize * sizeof(FpType);
    CUDA_CHECK(cudaFuncSetAttribute(batched_lu_subwarp<FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices>, cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    batched_lu_subwarp<FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices><<<numBlocks, numThreads, shMemSize>>>(d_A);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    CUDA_CHECK(cudaMemcpy(A_inv.data(), d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost));
    std::cout << "Data copied back to host." << '\n';

    // print A_inv
    // printf("\n");
    // printMatrices(A, matrixSize, numMatrices);
    // printf("\n");
    // printMatrices(A_inv, matrixSize, numMatrices);

    // write A_inv to file
    // auto filename = "8thread32_mtrx.txt";
    // writeToFile(A_inv, filename, matrixSize, numMatrices);

    startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    // verifyLU(A, A_inv, matrixSize, numMatrices);
    endT = std::chrono::high_resolution_clock::now();
    elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";

    cudaFree(d_A);

    return 0;
}
