#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using FpType = float;

#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }
#define CUBLAS_CHECK(call) { cublasStatus_t status = call; if (status != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error: %d, line %d\n", status, __LINE__); exit(1); } }

// Helper function to initialize a matrix with random values
void initializeMatrix(FpType* matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(static_cast<FpType>(-1.0), static_cast<FpType>(1.0));
    std::ifstream file("matrix.txt");
    for (int i = 0; i < n * n; ++i) {
        file >> matrix[i];
        // matrix[i] = static_cast<FpType>(dis(gen));
    }
}

int main() {

    int num_runs = 10;
    std::vector<float> run_times(num_runs);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    constexpr int BATCH_SIZE=INP_BATCH_SIZE;
    constexpr int N=INP_MATRIX_SIZE;

    // Host memory
    std::vector<FpType*> h_A(BATCH_SIZE);
    std::vector<FpType*> h_invA(BATCH_SIZE);

    std::cout << "Batch size: " << BATCH_SIZE << '\n';
    std::cout << "Matrix size: " << N << '\n';

    std::cout << "Initializing matrices..." << std::endl;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        h_A[i] = new FpType[N * N];
        h_invA[i] = new FpType[N * N];
        initializeMatrix(h_A[i], N);
        // init input matrix as 0.0
        for (int j = 0; j < N * N; ++j) {
            h_invA[i][j] = static_cast<FpType>(0.0);
        }
    }

    std::cout << "Matrices initialized." << std::endl;

    for (int rt = 0; rt < num_runs; rt++) {

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
        CUDA_CHECK(cudaMemcpy(d_invA[i], h_invA[i], N * N * sizeof(FpType), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_A_array, d_A.data(), BATCH_SIZE * sizeof(FpType*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invA_array, d_invA.data(), BATCH_SIZE * sizeof(FpType*), cudaMemcpyHostToDevice));

    std::cout << "Matrices copied to device." << std::endl;

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

    run_times[rt] = milliseconds;

    // Copy results back to host i.e all the matrices
    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUDA_CHECK(cudaMemcpy(h_invA[i], d_invA[i], N * N * sizeof(FpType), cudaMemcpyDeviceToHost));
    }

    std::cout << "Data copied back to host." << std::endl;

    std::cout << "Verifying inversion..." << std::endl;

    int diagCnt = 0, offDiagCnt = 0;
    int corrInv = 0, wrongInv = 0;

    FpType *A = new FpType[N * N];
    FpType *A_inv = new FpType[N * N];

    for (int i = 0; i < BATCH_SIZE; ++i) {
        FpType *A = h_A[i];
        FpType *invA = h_invA[i];
        FpType *result = new FpType[N * N];

        diagCnt = 0;
        offDiagCnt = 0;

        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                FpType sum = 0.0;
                for (int l = 0; l < N; ++l) {
                    sum += A[j * N + l] * invA[l * N + k];
                }
                result[j * N + k] = sum;
            }
        }

        // verify if the result is the identity matrix
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                if (j == k && std::abs(result[j * N + k] - FpType(1.0)) < 1e-3) {
                    diagCnt++;
                } else if (j != k && std::abs(result[j * N + k] - FpType(0.0)) < 1e-3) {
                    offDiagCnt++;
                }
            }
        }
        
        if (diagCnt == N && offDiagCnt == N * (N - 1)) {
            corrInv++;
        } else {
            wrongInv++;
        }

        delete[] result;
    }

    std::cout << "Correctly inverted matrices: " << corrInv << std::endl;
    std::cout << "Incorrectly inverted matrices: " << wrongInv << std::endl;

    // Clean up

    delete[] A;
    delete[] A_inv;

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

    }
    // write the run times to a file based on matrix size
    std::ofstream file;
    auto filename = "cublas_" + std::to_string(N) + ".csv";
    file.open(filename);
    for (int i = 0; i < num_runs; i++) {
        file << run_times[i] << std::endl;
    }
    file.close();

    return 0;
}