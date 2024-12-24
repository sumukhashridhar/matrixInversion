#include "luBatchedInplace.cuh"

int main() {
    constexpr int matrixSize=MATRIXSIZE;
    constexpr int numMatrices=NUMMATRICES;
    constexpr int numThreads=NUMTHREADS;

    // const int threadsPerMatrix = matrixSize;
    // // const int threadsPerMatrix = 32;
    // const int matricesPerBlock = numThreads / threadsPerMatrix;
    // const int numBlocks = numMatrices / matricesPerBlock;

    constexpr int threadsPerMatrix = matrixSize;
    // constexpr int threadsPerMatrix = 32;
    constexpr int matricesPerBlock = numThreads / threadsPerMatrix;
    constexpr int numBlocks = numMatrices / matricesPerBlock;

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

    // int numElements = matrixSize * matrixSize * numMatrices;
    constexpr int numElements = matrixSize * matrixSize * numMatrices;

    std::vector<FpType> A(numElements);
    std::vector<FpType> A_inv(numElements);

    std::cout << "Reading data from file." << '\n';

        #pragma acc parallel loop
        for (int k = 0; k < numMatrices; ++k) {
            int offset = k * matrixSize * matrixSize;
            // std::ifstream file("main_mtrx.txt");
            std::ifstream file("matrix.txt");
            
            #pragma acc loop vector
            for (int i = 0; i < matrixSize; ++i) {
                for (int j = 0; j < matrixSize; ++j) {
                    // float temp;
                    // file >> temp;
                    // A[(i * matrixSize) + offset + j] = __float2half(temp);
                    file >> A[(i * matrixSize) + offset + j];
                    A_inv[(i * matrixSize) + offset + j] = static_cast<FpType>(0.0);
                }
            }
            file.close();
        }

    std::cout << "Data read from file." << '\n';

    // int num_runs = 10;
    // std::vector<float> run_times(num_runs);

    // for (int rt = 0; rt < num_runs; rt++) {
    FpType* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, numElements * sizeof(FpType)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), numElements * sizeof(FpType), cudaMemcpyHostToDevice));
    std::cout << "Data copied to device." << '\n';

    // int shMemSize = matricesPerBlock * matrixSize * matrixSize * sizeof(FpType);
    constexpr int shMemSize = matricesPerBlock * matrixSize * matrixSize * sizeof(FpType);
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
    // run_times[rt] = milliseconds;

    CUDA_CHECK(cudaMemcpy(A_inv.data(), d_A, numElements * sizeof(FpType), cudaMemcpyDeviceToHost));
    std::cout << "Data copied back to host." << '\n';

    auto startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    // verifyLU(A, A_inv, matrixSize, numMatrices);
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    // }

    // // write the run times to a file based on matrix size
    // std::ofstream file;
    // auto filename = "cublas_" + std::to_string(N) + ".csv";
    // file.open(filename);
    // for (int i = 0; i < num_runs; i++) {
    //     file << run_times[i] << std::endl;
    // }
    // file.close();


    return 0;
}
