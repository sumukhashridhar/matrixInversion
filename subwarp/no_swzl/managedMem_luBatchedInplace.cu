#include "managedMem_luBatchedInplace.cuh"

int main() {
    constexpr int matrixSize = MATRIXSIZE;
    constexpr int numMatrices = NUMMATRICES;
    constexpr int numThreads = NUMTHREADS;

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

    // Allocate managed memory for A and A_inv
    FpType* A = nullptr;
    FpType* A_inv = nullptr;
    CUDA_CHECK(cudaMallocManaged(&A, numElements * sizeof(FpType)));
    CUDA_CHECK(cudaMallocManaged(&A_inv, numElements * sizeof(FpType)));

    std::cout << "Reading data from file." << '\n';

    // Initialize data
    #pragma acc parallel loop
    for (int k = 0; k < numMatrices; ++k) {
        int offset = k * matrixSize * matrixSize;
        std::ifstream file("matrix.txt");
        
        #pragma acc loop vector
        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                file >> A[(i * matrixSize) + offset + j];
                A_inv[(i * matrixSize) + offset + j] = static_cast<FpType>(0.0);
            }
        }
        file.close();
    }

    std::cout << "Data read from file." << '\n';

    // Prefetch data to GPU
    int device = -1;
    cudaGetDevice(&device);
    CUDA_CHECK(cudaMemPrefetchAsync(A, numElements * sizeof(FpType), device));
    CUDA_CHECK(cudaMemPrefetchAsync(A_inv, numElements * sizeof(FpType), device));

    int shMemSize = matricesPerBlock * matrixSize * matrixSize * sizeof(FpType);
    auto kernel = batched_lu_subwarp<FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize));

    // CUDA_CHECK(cudaFuncSetAttribute(batched_lu_subwarp<FpType, FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices>, 
                                //   cudaFuncAttributeMaxDynamicSharedMemorySize, shMemSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    batched_lu_subwarp<FpType, matrixSize, threadsPerMatrix, matricesPerBlock, numMatrices><<<numBlocks, numThreads, shMemSize>>>(A, A_inv);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    // Prefetch data back to CPU for verification
    CUDA_CHECK(cudaMemPrefetchAsync(A, numElements * sizeof(FpType), cudaCpuDeviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(A_inv, numElements * sizeof(FpType), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto startT = std::chrono::high_resolution_clock::now();
    verifyInv(A, A_inv, matrixSize, numMatrices);
    auto endT = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endT - startT;
    std::cout << "Time taken to verify inverse: " << elapsed.count() << " seconds\n";

    // Clean up
    cudaFree(A);
    cudaFree(A_inv);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}