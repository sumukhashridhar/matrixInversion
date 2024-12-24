#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <fstream>

// #include <cuda_fp16.h>

using FpType = float;
// using FpType = __half;

void printMatrices(const std::vector<FpType> A_inv, int matrixSize, int numMatrices)
{
    for (int k = 0; k < numMatrices; ++k)
    {
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                std::cout << (A_inv[(i * matrixSize) + (k * matrixSize * matrixSize) + j]) << ' ';
                // std::cout << __half2float (A_inv[(i * matrixSize) + (k * matrixSize * matrixSize) + j]) << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
        break;
    }
}

void writeToFile(const std::vector<FpType> A_inv, const std::string &filename, int matrixSize, int numMatrices)
{
    std::ofstream file(filename);
    for (int k = 0; k < numMatrices; ++k)
    {
        int offset = k * matrixSize * matrixSize;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                // float temp = __half2float(A_inv[(i * matrixSize) + offset + j]);
                // file << temp << " ";
                file << A_inv[(i * matrixSize) + offset + j] << " ";
            }
            file << '\n';
        }
        break;
    }
    file.close();
}

void verifyInv(const std::vector<FpType> A, const std::vector<FpType> A_inv, int matrixSize, int numMatrices)
{
    FpType result = static_cast<FpType>(0.0);
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
                result = static_cast<FpType>(0.0);
#pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += A[(j * matrixSize) + offset + l] * A_inv[(l * matrixSize) + offset + i];
                }

                if (i == j && std::fabs(result - FpType(1.0)) < 1e-3)
                // if (i == j && std::fabs(__half2float(result) - __half2float(1.0)) < 1e-3)
                {
                    idCnt++;
                    // std::cout << "Diff is " << std::fabs(result - FpType(1.0)) << " for i = " << i << " and j = " << j << '\n';
                }

                if (i != j && std::fabs(result - FpType(0.0)) < 1e-3)
                // if (i != j && std::fabs(__half2float(result) - __half2float(0.0)) < 1e-3)
                {
                    offDiagCnt++;
                    // std::cout << "Diff is " << std::fabs(result - FpType(1.0)) << " for i = " << i << " and j = " << j << '\n';
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

void verifyLU(const std::vector<FpType> A, std::vector<FpType> LU, int matrixSize, int numMatrices)
{
    // store L and U matrices separately from LU
    std::vector<FpType> L(matrixSize * matrixSize * numMatrices, 0.0);
    std::vector<FpType> U(matrixSize * matrixSize * numMatrices, 0.0);

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

    FpType result = static_cast<FpType>(0.0);
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
                result = static_cast<FpType>(0.0);
#pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += L[(i * matrixSize) + offset + l] * U[(l * matrixSize) + offset + j];
                }
                LU[(i * matrixSize) + offset + j] = result;

                if (std::fabs(A[(i * matrixSize) + offset + j] - result) < 1e-3)
                // if (std::fabs(__half2float(A[(i * matrixSize) + offset + j]) - __half2float(result)) < 1e-3)
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
