#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>

using FpType = float;

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }
#define CUBLAS_CHECK(call) { cublasStatus_t status = call; if (status != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error: %d, line %d\n", status, __LINE__); exit(1); } }

// const int BATCH_SIZE = 1000000;
// const int N = 8; // 8x8 matrices

// Helper function to initialize a matrix with random values
void initializeMatrix(FpType* matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(static_cast<FpType>(-1.0), static_cast<FpType>(1.0));
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<FpType>(dis(gen));
    }
}

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int BATCH_SIZE, N;

    std::cout << "Enter batch size: ";
    std::cin >> BATCH_SIZE;
    std::cout << "Enter matrix size: ";
    std::cin >> N;

    // Host memory
    std::vector<FpType*> h_A(BATCH_SIZE);
    std::vector<FpType*> h_invA(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        h_A[i] = new FpType[N * N];
        h_invA[i] = new FpType[N * N];
        initializeMatrix(h_A[i], N);
    }

    // Device memory
    FpType **d_A_array, **d_invA_array;
    int *d_pivots, *d_info;
    CUDA_CHECK(cudaMalloc(&d_A_array, BATCH_SIZE * sizeof(FpType*)));
    CUDA_CHECK(cudaMalloc(&d_invA_array, BATCH_SIZE * sizeof(FpType*)));
    CUDA_CHECK(cudaMalloc(&d_pivots, BATCH_SIZE * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, BATCH_SIZE * sizeof(int)));

    std::vector<FpType*> d_A(BATCH_SIZE);
    std::vector<FpType*> d_invA(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUDA_CHECK(cudaMalloc(&d_A[i], N * N * sizeof(FpType)));
        CUDA_CHECK(cudaMalloc(&d_invA[i], N * N * sizeof(FpType)));
        CUDA_CHECK(cudaMemcpy(d_A[i], h_A[i], N * N * sizeof(FpType), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_A_array, d_A.data(), BATCH_SIZE * sizeof(FpType*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invA_array, d_invA.data(), BATCH_SIZE * sizeof(FpType*), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Perform batched LU factorization
    CUBLAS_CHECK(cublasSgetrfBatched(handle, N, d_A_array, N, d_pivots, d_info, BATCH_SIZE));

    // Perform batched inversion
    CUBLAS_CHECK(cublasSgetriBatched(handle, N, (const FpType **)d_A_array, N, d_pivots,
                                     d_invA_array, N, d_info, BATCH_SIZE));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "cuBLAS Kernel execution time: " << milliseconds << " milliseconds\n";

    // Copy results back to host (just the first matrix for demonstration)
    CUDA_CHECK(cudaMemcpy(h_invA[0], d_invA[0], N * N * sizeof(FpType), cudaMemcpyDeviceToHost));

    // Print the first inverted matrix
    // std::cout << "First inverted matrix:" << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << h_invA[0][i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));

    for (int i = 0; i < BATCH_SIZE; ++i) {
        delete[] h_A[i];
        delete[] h_invA[i];
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_invA[i]));
    }
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_invA_array));
    CUDA_CHECK(cudaFree(d_pivots));
    CUDA_CHECK(cudaFree(d_info));

    return 0;
}