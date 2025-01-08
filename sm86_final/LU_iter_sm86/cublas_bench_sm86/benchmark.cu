#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using FpType = float;

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } }
#define CUBLAS_CHECK(call) { cublasStatus_t status = call; if (status != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error: %d (%s:%d)\n", status, __FILE__, __LINE__); exit(1); } }

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    constexpr int BATCH_SIZE = INP_BATCH_SIZE;
    constexpr int N = INP_MATRIX_SIZE;

    std::cout << "Matrix size: " << N << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;

    std::vector<FpType*> h_A(BATCH_SIZE);
    std::vector<FpType*> h_L(BATCH_SIZE);
    std::vector<FpType*> h_U(BATCH_SIZE);

    // Read template matrix
    std::vector<FpType> templateMatrix(N * N);
    {
        std::ifstream file("mtrand32_new1.txt");
        if (!file.is_open()) {
            std::cerr << "Failed to open the file" << std::endl;
            return 1;
        }
        for (int i = 0; i < N * N; ++i) {
            file >> templateMatrix[i];
        }
    }

    // Initialize host matrices with pinned memory
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUDA_CHECK(cudaHostAlloc(&h_A[i], N * N * sizeof(FpType), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_L[i], N * N * sizeof(FpType), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_U[i], N * N * sizeof(FpType), cudaHostAllocDefault));
        
        #pragma omp simd
        for (int j = 0; j < N * N; ++j) {
            h_A[i][j] = templateMatrix[j];
            h_L[i][j] = 0.0f;
            h_U[i][j] = 0.0f;
        }
    }

    // Setup CUDA streams
    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory
    FpType **d_A_array;
    int *d_pivots, *d_info;
    CUDA_CHECK(cudaMalloc(&d_A_array, BATCH_SIZE * sizeof(FpType*)));
    CUDA_CHECK(cudaMalloc(&d_pivots, BATCH_SIZE * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, BATCH_SIZE * sizeof(int)));

    std::vector<FpType*> d_A(BATCH_SIZE);
    
    // Transfer data using streams
    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUDA_CHECK(cudaMalloc(&d_A[i], N * N * sizeof(FpType)));
        int streamIdx = i % NUM_STREAMS;
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], h_A[i], N * N * sizeof(FpType), 
                                  cudaMemcpyHostToDevice, streams[streamIdx]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_A_array, d_A.data(), BATCH_SIZE * sizeof(FpType*), cudaMemcpyHostToDevice));

    // Perform batched LU factorization
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CUBLAS_CHECK(cublasSgetrfBatched(handle, N, d_A_array, N, d_pivots, d_info, BATCH_SIZE));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "cuBLAS Kernel execution time: " << milliseconds << " milliseconds\n";


    // Copy results back
    for (int i = 0; i < BATCH_SIZE; ++i) {
        int streamIdx = i % NUM_STREAMS;
        CUDA_CHECK(cudaMemcpyAsync(h_L[i], d_A[i], N * N * sizeof(FpType), 
                                  cudaMemcpyDeviceToHost, streams[streamIdx]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify LU decomposition
    std::cout << "Verifying LU decomposition..." << std::endl;
    int correctDecomp = 0, incorrectDecomp = 0;
    constexpr FpType TOLERANCE = 1e-3f;

    #pragma omp parallel for reduction(+:correctDecomp) schedule(dynamic)
    for (int b = 0; b < BATCH_SIZE; ++b) {
        // Extract L and U from the result
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i > j) {
                    h_L[b][i * N + j] = h_A[b][i * N + j];
                    h_U[b][i * N + j] = 0.0f;
                } else if (i == j) {
                    h_L[b][i * N + j] = 1.0f;
                    h_U[b][i * N + j] = h_A[b][i * N + j];
                } else {
                    h_L[b][i * N + j] = 0.0f;
                    h_U[b][i * N + j] = h_A[b][i * N + j];
                }
            }
        }

        // Verify L*U = A
        bool isCorrect = true;
        std::vector<FpType> result(N * N);
        
        #pragma omp simd collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                FpType sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += h_L[b][i * N + k] * h_U[b][k * N + j];
                }
                result[i * N + j] = sum;
                if (std::abs(result[i * N + j] - templateMatrix[i * N + j]) >= TOLERANCE) {
                    isCorrect = false;
                }
            }
        }
        
        if (isCorrect) correctDecomp++;
        else (!isCorrect) incorrectDecomp++;
    }

    std::cout << "Correct inversions: " << correctDecomp << << std::endl;
    std::cout << "Incorrect inversions: " << incorrectDecomp << std::endl;

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_pivots));
    CUDA_CHECK(cudaFree(d_info));

    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUDA_CHECK(cudaFreeHost(h_A[i]));
        CUDA_CHECK(cudaFreeHost(h_L[i]));
        CUDA_CHECK(cudaFreeHost(h_U[i]));
        CUDA_CHECK(cudaFree(d_A[i]));
    }

    return 0;
}