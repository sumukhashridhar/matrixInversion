#include "luBatchedInplace.cuh"

int main() {
    constexpr int matrixSize=MATRIXSIZE;
    constexpr int numMatrices=NUMMATRICES;
    constexpr int numThreads=NUMTHREADS;

    constexpr int threadsPerMatrix = matrixSize;
    // const int threadsPerMatrix = 32;
    constexpr int matricesPerBlock = numThreads / threadsPerMatrix;
    constexpr int numBlocks = numMatrices / matricesPerBlock + (numMatrices % matricesPerBlock == 0 ? 0 : 1);

    std::cout << "Matrix size: " << matrixSize << '\n';
    std::cout << "Number of matrices: " << numMatrices << '\n'; 
    std::cout << "Number of threads per block: " << numThreads << '\n';
    std::cout << "Threads per matrix: " << threadsPerMatrix << '\n';
    std::cout << "Matrices per block: " << matricesPerBlock << '\n';
    std::cout << "Number of blocks: " << numBlocks << '\n';

    // FpType inputMatrix[] = {2, 3, 4, 5};//, 10, 4, 2, 4, 2};
    //  FpType inputMatrix[] = {4, 11, 3, 40, 10, 4, 2, 40, 2};
    FpType inputMatrix[] = {4, 11, 3, 7, 4, 10, 4, 9, 2, 4, 2, 1, 7, 9, 18, 30};
    // FpType inputMatrix[] = {2, 7, 1, 5, 3, -2, 0, 1, 1, 5, 3, 4, 7, 3, 2, 8};
    // make input matrix as Identity matrix
    // FpType inputMatrix[] = {10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10};

    constexpr int numElements = matrixSize * matrixSize * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<FpType> dist(static_cast<FpType>(0.0), static_cast<FpType>(1.0));
    // std::cout << "Generating random data." << '\n';

    std::cout << "Reading data from file." << '\n';

    // Read the file once into a single matrix-sized buffer
    // std::ifstream file("matrix.txt");
    // std::ifstream file("mtrand32_new.txt");
    // std::ifstream file("mtrand32.txt");
    std::ifstream file("mtrand32_new1.txt");
    std::vector<FpType> templateMatrix(matrixSize * matrixSize);
    for (int i = 0; i < matrixSize * matrixSize; ++i) {
        file >> templateMatrix[i];
        // templateMatrix[i] = inputMatrix[i];
        // use random values
        // templateMatrix[i] = rand() % 10;
        // templateMatrix[i] = dist(engine);
    }
    file.close();

    // printMatrices(templateMatrix, matrixSize, 1);

    // // fill the matrix with upper triangular values
    // for (int i = 0; i < matrixSize; ++i) {
    //     for (int j = 0; j < matrixSize; ++j) {
    //         if (j > i) {
    //             templateMatrix[(i * matrixSize) + j] = static_cast<FpType>(0.0);
    //         }
    //         if (i == j) {
    //             templateMatrix[(i * matrixSize) + j] = static_cast<FpType>(1.0);
    //         }
    //     }
    // }

    // print the input matrix
    // printMatrices(templateMatrix, matrixSize, 1);

    std::vector<int> pivots(matrixSize);
    // fill the pivots with 0
    for (int i = 0; i < matrixSize; ++i) {
        pivots[i] = 0;
    }

    // save the matrix to filename mtrand32.txt
    // writeToFile(templateMatrix, "mtrand32_new1.txt", matrixSize, 1);

    FpType cond_num = calc_cond_num(templateMatrix, matrixSize);
    std::cout << "Condition number of the matrix is: " << cond_num << '\n';

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

    // // print the pivots
    // for (int i = 0; i < matrixSize; ++i) {
    //     std::cout << pivots[i] << "\n";
    // }

    FpType* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, numElements * sizeof(FpType)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), numElements * sizeof(FpType), cudaMemcpyHostToDevice));
    std::cout << "Data copied to device." << '\n';

    // constexpr int shMemSize = matricesPerBlock * matrixSize * matrixSize * sizeof(FpType);
    constexpr int shMemSize = matricesPerBlock * (matrixSize * matrixSize + matrixSize + 2 * threadsPerMatrix) * sizeof(FpType);
    // constexpr int shMemSize = matricesPerBlock * (matrixSize * matrixSize + matrixSize) * sizeof(FpType);
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

    std::vector<FpType> vec_pivotedA(matrixSize * matrixSize);
    pivotedA(templateMatrix, vec_pivotedA, pivots, matrixSize);

    // print the pivots
    // for (int i = 0; i < matrixSize; ++i) {
    //     std::cout << pivots[i] << "\n";
    // }


    startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    // verifyLUwithPivoting(vec_pivotedA, A_inv, matrixSize, numMatrices);
    endT = std::chrono::high_resolution_clock::now();
    elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";

    cudaFree(d_A);

    return 0;
}
