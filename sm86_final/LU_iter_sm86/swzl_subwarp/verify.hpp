#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <fstream>
#include <random>

// using FpType = double;
using FpType = float;

template <typename T>
void printMatrices(const std::vector<T> A_inv, int matrixSize, int numMatrices)
{
    for (int k = 0; k < numMatrices; ++k)
    {
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                std::cout << (A_inv[(i * matrixSize) + (k * matrixSize * matrixSize) + j]) << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
        break;
    }
}

template <typename T>
void writeToFile(const std::vector<T> A_inv, const std::string &filename, int matrixSize, int numMatrices)
{
    std::ofstream file(filename);
    for (int k = 0; k < numMatrices; ++k)
    {
        int offset = k * matrixSize * matrixSize;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                file << A_inv[(i * matrixSize) + offset + j] << " ";
            }
            file << '\n';
        }
        break;
    }
    file.close();
}

template <typename T>
void verifyInv(const std::vector<T> A, const std::vector<T> A_inv, int matrixSize, int numMatrices)
{
    T result = static_cast<T>(0.0), threshold = static_cast<T>(1e-3);
    int corrInv = 0, wrngInv = 0, offset = 0, idCnt = 0, offDiagCnt = 0;

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; ++k)
    {
        offset = k * matrixSize * matrixSize;
        idCnt = 0;
        offDiagCnt = 0;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                result = static_cast<T>(0.0);
#pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += A[(j * matrixSize) + offset + l] * A_inv[(l * matrixSize) + offset + i];
                }

                if (i == j && std::fabs(result - T(1.0)) < threshold)
                {
                    idCnt++;
                    // std::cout << "Diff is " << std::fabs(result - T(1.0)) << " for i = " << i << " and j = " << j << '\n';
                }

                if (i != j && std::fabs(result - T(0.0)) < threshold)
                {
                    offDiagCnt++;
                    // std::cout << "Diff is " << std::fabs(result - T(1.0)) << " for i = " << i << " and j = " << j << '\n';
                    // print result
                    // std::cout << "Result is " << result << '\n';
                }
            }
        }
        if (idCnt == matrixSize && offDiagCnt == matrixSize * (matrixSize - 1))
        {
            corrInv++;
        }
        else
        {
            wrngInv++;
        }
    }

    // std::cout << "Identity elements: " << idCnt << '\n';
    // std::cout << "Off-diagonal elements: " << offDiagCnt << '\n';

    std::cout << "Correct inversions: " << corrInv << '\n';
    std::cout << "Incorrect inversions: " << wrngInv << '\n';
}

template <typename T>
void verifyLU(const std::vector<T> A, std::vector<T> LU, int matrixSize, int numMatrices)
{
    // store L and U matrices separately from LU
    std::vector<T> L(matrixSize * matrixSize * numMatrices, 0.0);
    std::vector<T> U(matrixSize * matrixSize * numMatrices, 0.0);

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; k++)
    {
        for (int i = 0; i < matrixSize; i++)
        {
            for (int j = 0; j < matrixSize; j++)
            {
                if (i == j)
                {
                    L[(i * matrixSize) + (k * matrixSize * matrixSize) + j] = 1.0;
                }

                if (i > j)
                {
                    L[(i * matrixSize) + (k * matrixSize * matrixSize) + j] = LU[(i * matrixSize) + (k * matrixSize * matrixSize) + j];
                }
            }
        }
    }

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; k++)
    {
        for (int i = 0; i < matrixSize; i++)
        {
            for (int j = 0; j < matrixSize; j++)
            {
                if (i <= j)
                {
                    U[(i * matrixSize) + (k * matrixSize * matrixSize) + j] = LU[(i * matrixSize) + (k * matrixSize * matrixSize) + j];
                }
            }
        }
    }

    T result = static_cast<T>(0.0), threshold = static_cast<T>(1e-3);
    int corrLU = 0, wrngLU = 0, offset = 0, idCnt = 0;

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; k++)
    {
        offset = k * matrixSize * matrixSize;
        idCnt = 0;
        for (int i = 0; i < matrixSize; i++)
        {
            for (int j = 0; j < matrixSize; j++)
            {
                result = static_cast<T>(0.0);
#pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += L[(i * matrixSize) + offset + l] * U[(l * matrixSize) + offset + j];
                }
                LU[(i * matrixSize) + offset + j] = result;

                if (std::fabs(A[(i * matrixSize) + offset + j] - result) < threshold)
                {
                    idCnt++;
                }
            }
        }

        if (idCnt == matrixSize * matrixSize)
        {
            corrLU++;
        }
        else
        {
            wrngLU++;
        }
    }

    std::cout << "Correct LU decompositions: " << corrLU << '\n';
    std::cout << "Incorrect LU decompositions: " << wrngLU << '\n';
}

template <typename T>
T calc_cond_num(const std::vector<T> &matrix, int N)
{
    // Helper function to get matrix element at (i,j)
    auto at = [N](const std::vector<T> &m, int i, int j) -> T
    {
        return m[i * N + j];
    };

    // Calculate L1 norm (maximum absolute column sum)
    T matrix_norm = 0;
    for (int j = 0; j < N; ++j)
    {
        T col_sum = 0;
        for (int i = 0; i < N; ++i)
        {
            col_sum += std::abs(at(matrix, i, j));
        }
        matrix_norm = std::max(matrix_norm, col_sum);
    }

    // Create a copy of the matrix for inversion
    std::vector<T> inv_matrix = matrix;

    // Gauss-Jordan elimination with pivoting
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i)
        p[i] = i;

    for (int i = 0; i < N; ++i)
    {
        // Find pivot
        T max_val = std::abs(at(inv_matrix, i, i));
        int max_idx = i;
        for (int j = i + 1; j < N; ++j)
        {
            T val = std::abs(at(inv_matrix, j, i));
            if (val > max_val)
            {
                max_val = val;
                max_idx = j;
            }
        }

        if (max_val < std::numeric_limits<T>::epsilon())
        {
            return std::numeric_limits<T>::infinity(); // Matrix is singular
        }

        // Swap rows if necessary
        if (max_idx != i)
        {
            for (int j = 0; j < N; ++j)
            {
                std::swap(inv_matrix[i * N + j], inv_matrix[max_idx * N + j]);
            }
            std::swap(p[i], p[max_idx]);
        }

        // Scale the pivot row
        T pivot = at(inv_matrix, i, i);
        for (int j = 0; j < N; ++j)
        {
            inv_matrix[i * N + j] /= pivot;
        }

        // Eliminate column
        for (int j = 0; j < N; ++j)
        {
            if (j != i)
            {
                T factor = at(inv_matrix, j, i);
                for (int k = 0; k < N; ++k)
                {
                    inv_matrix[j * N + k] -= factor * at(inv_matrix, i, k);
                }
            }
        }
    }

    // Calculate L1 norm of inverse matrix
    T inv_norm = 0;
    for (int j = 0; j < N; ++j)
    {
        T col_sum = 0;
        for (int i = 0; i < N; ++i)
        {
            col_sum += std::abs(at(inv_matrix, i, j));
        }
        inv_norm = std::max(inv_norm, col_sum);
    }

    return matrix_norm * inv_norm;
}
