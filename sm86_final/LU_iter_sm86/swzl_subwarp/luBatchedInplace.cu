#include "luBatchedInplace.cuh"

int main() {
    constexpr int matrixSize=MATRIXSIZE;
    constexpr int numMatrices=NUMMATRICES;
    constexpr int numThreads=NUMTHREADS;

    // round of the matrixSize to nearest power of 2
    constexpr int matrixSizePow2 = 1 << (32 - __builtin_clz(matrixSize - 1));

    constexpr int threadsPerMatrix = matrixSizePow2;
    constexpr int matricesPerBlock = numThreads / threadsPerMatrix;
    constexpr int numBlocks = numMatrices / matricesPerBlock + (numMatrices % matricesPerBlock == 0 ? 0 : 1);

    std::cout << "Matrix size: " << matrixSize << '\n';
    std::cout << "Number of matrices: " << numMatrices << '\n'; 
    std::cout << "Number of threads per block: " << numThreads << '\n';
    std::cout << "Threads per matrix: " << threadsPerMatrix << '\n';
    std::cout << "Matrices per block: " << matricesPerBlock << '\n';
    std::cout << "Number of blocks: " << numBlocks << '\n';

    constexpr int numElements = matrixSizePow2 * matrixSizePow2 * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    std::cout << "Reading data from file." << '\n';

    // Read the file once into a single matrix-sized buffer
    // std::ifstream file("matrix.txt");
    // std::ifstream file ("mtrand32.txt");
    std::ifstream file ("mtrand32_new1.txt");

    // create a matrix with size matrixSizePow2 x matrixSizePow2, but only fill the first matrixSize x matrixSize elements and among the rest fill the diagonal with 1s and the rest with 0s
    std::vector<FpType> templateMatrix(matrixSizePow2 * matrixSizePow2);
    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            file >> templateMatrix[(i * matrixSizePow2) + j];
        }
    }

    file.close();
    for (int i = matrixSize; i < matrixSizePow2; ++i) {
        for (int j = 0; j < matrixSizePow2; ++j) {
            if (j == i) {
                templateMatrix[(i * matrixSizePow2) + j] = static_cast<FpType>(1.0);
            } else {
                templateMatrix[(i * matrixSizePow2) + j] = static_cast<FpType>(0.0);
            }
        }
    }

    // printMatrices(templateMatrix, matrixSize, 1);

    FpType cond_num = calc_cond_num(templateMatrix, matrixSizePow2);
    std::cout << "Condition number: " << cond_num << '\n';

    auto startT = std::chrono::high_resolution_clock::now();
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;

    startT = std::chrono::high_resolution_clock::now();
    #pragma acc parallel loop
    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSizePow2 * matrixSizePow2;
        
        #pragma acc loop vector
        for (int i = 0; i < matrixSizePow2; ++i) {
            for (int j = 0; j < matrixSizePow2; ++j) {
                int templateIndex = (i * matrixSizePow2) + j;
                A[(i * matrixSizePow2) + offset + j] = templateMatrix[templateIndex];
                A_inv[(i * matrixSizePow2) + offset + j] = static_cast<FpType>(0.0);
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

    int shMemSize = matricesPerBlock * matrixSizePow2 * matrixSizePow2 * sizeof(FpType);
    CUDA_CHECK(cudaFuncSetAttribute(batched_lu_subwarp<FpType, matrixSizePow2, threadsPerMatrix, matricesPerBlock, numMatrices>, cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    batched_lu_subwarp<FpType, matrixSizePow2, threadsPerMatrix, matricesPerBlock, numMatrices><<<numBlocks, numThreads, shMemSize>>>(d_A);
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
    // verifyInv(A, A_inv, matrixSizePow2, numMatrices);
    verifyLU(A, A_inv, matrixSize, numMatrices);
    endT = std::chrono::high_resolution_clock::now();
    elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";

    cudaFree(d_A);

    return 0;
}
