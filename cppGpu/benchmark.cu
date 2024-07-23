#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper function to check for CUDA errors
void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Helper function to check for cuBLAS errors
void checkCublas(cublasStatus_t result, const char *msg) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": " << result << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int N = 4; // Matrix dimension (N x N)
    int batchSize = 2; // Number of matrices in the batch
    int numElements = N * N;

    // // // Host input matrices
    float A[N * N] = {
        4, 11, 3,
        4, 10, 4,
        2, 4, 2,
    };

    // create a random input matrix of size N x N
    // float A[N * N];
    // for (int i = 0; i < N * N; i++) {
    //     A[i] = rand() % 10;
    // }

    // std::vector<float> A(numElements);
    // std::ifstream file("matrix.txt");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         file >> A[i * N + j];
    //     }
    // }
    // file.close();

    float h_A[batchSize][N * N];

    // Copy the input matrix to the batch
    for (int i = 0; i < batchSize; i++) {
        memcpy(h_A[i], A, N * N * sizeof(float));
    }

    // Host output matrices (for the results)
    float h_Ainv[batchSize][N * N];

    // Device arrays
    float *d_A;
    float **d_Aarray;
    int *d_infoArray;
    int *h_infoArray = new int[batchSize];

    // Allocate device memory
    checkCuda(cudaMalloc((void **)&d_A, batchSize * N * N * sizeof(float)), "Failed to allocate device memory for A");
    checkCuda(cudaMalloc((void **)&d_Aarray, batchSize * sizeof(float *)), "Failed to allocate device memory for A array");
    checkCuda(cudaMalloc((void **)&d_infoArray, batchSize * sizeof(int)), "Failed to allocate device memory for info array");

    // Copy host data to device
    checkCuda(cudaMemcpy(d_A, h_A, batchSize * N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy A to device");

    // Create array of pointers
    std::vector<float *> h_Aptr(batchSize);
    for (int i = 0; i < batchSize; i++) {
        h_Aptr[i] = d_A + i * N * N;
    }
    checkCuda(cudaMemcpy(d_Aarray, h_Aptr.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice), "Failed to copy A array to device");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize cuBLAS
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Start timer
    cudaEventRecord(start, 0);

    // Perform LU decomposition
    checkCublas(cublasSgetrfBatched(handle, N, d_Aarray, N, nullptr, d_infoArray, batchSize), "Failed to perform LU decomposition");

    // // Check for singular matrices
    // checkCuda(cudaMemcpy(h_infoArray, d_infoArray, batchSize * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy info array to host");
    // for (int i = 0; i < batchSize; i++) {
    //     if (h_infoArray[i] != 0) {
    //         std::cerr << "Matrix " << i << " is singular" << std::endl;
    //         exit(EXIT_FAILURE);
    //     }
    // }

    // Perform matrix inversion
    checkCublas(cublasSgetriBatched(handle, N, d_Aarray, N, nullptr, d_Aarray, N, d_infoArray, batchSize), "Failed to perform matrix inversion");

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cublasTime = 0;
    cudaEventElapsedTime(&cublasTime, start, stop);
    std::cout << "cuBLAS time: " << cublasTime << " milliseconds\n";

    // Copy results back to host
    checkCuda(cudaMemcpy(h_Ainv, d_A, batchSize * N * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy Ainv to host");

    // Clean up
    checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");
    checkCuda(cudaFree(d_A), "Failed to free device memory for A");
    checkCuda(cudaFree(d_Aarray), "Failed to free device memory for A array");
    checkCuda(cudaFree(d_infoArray), "Failed to free device memory for info array");
    delete[] h_infoArray;

    // Print the results
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Inverted matrix " << b << ":\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_Ainv[b][i * N + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    int corrInv = 0, wrongInv = 0;

    // Verify the results
    #pragma omp parallel for collapse(1)
 for (int b = 0; b < batchSize; b++) {
            int idCnt = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float result = 0.0;
                    #pragma omp simd reduction(+ : result)
                    for (int k = 0; k < N; k++) {
                        result += h_A[b][i * N + k] * h_Ainv[b][k * N + j];
                    }
                    if (i == j && std::fabs(result - 1.0) < 1e-6)
                    {
                        idCnt++;
                    }
                }
            }
            if (idCnt == N) {
                corrInv++;
            } else {
                wrongInv++;
            }
        }

    std::cout << "Correct inversions: " << corrInv << "\n";
    std::cout << "Wrong inversions: " << wrongInv << "\n";


    return 0;
}
