#include <iostream>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>

using FpType = float;

void verifyInv(const std::vector<FpType> A, const std::vector<FpType> A_inv, int matrixSize, int numMatrices)
{
    FpType result = static_cast<FpType>(0.0);
    int corrInv = 0, wrngInv = 0, offset = 0, idCnt = 0;

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; ++k)
    {
        offset = k * matrixSize * matrixSize;
        idCnt = 0;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                result = static_cast<FpType>(0.0);
#pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += A[(i * matrixSize) + offset + l] * A_inv[(l * matrixSize) + offset + j];
                }
                // std::cout << result << ' ';
                if (i == j && std::fabs(result - FpType(1.0)) < 1e-3)
                {
                    idCnt++;
                }
            }
            // std::cout << '\n';
        }
        // std::cout << "Matrix " << k << " has " << idCnt << " identity elements\n";
        if (idCnt == matrixSize)
        {
            corrInv++;
        }
        else
        {
            wrngInv++;
        }
    }

    std::cout << "Correct inversions: " << corrInv << '\n';
    std::cout << "Incorrect inversions: " << wrngInv << '\n';
}


void verifyLU(FpType *A, int matrixSize, int numMatrices)
{
    FpType result = static_cast<FpType>(0.0);
    int corrLU = 0, wrngLU = 0, offset = 0, idCnt = 0;

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; ++k)
    {
        offset = k * matrixSize * matrixSize;
        idCnt = 0;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                result = static_cast<FpType>(0.0);
#pragma omp simd reduction(+ : result)
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += A[(i * matrixSize) + offset + l] * A[(l * matrixSize) + offset + j];
                }
                // std::cout << result << ' ';
                if (i == j && std::fabs(result - FpType(1.0)) < 1e-3)
                {
                    idCnt++;
                }
            }
            // std::cout << '\n';
        }
        // std::cout << "Matrix " << k << " has " << idCnt << " identity elements\n";
        if (idCnt == matrixSize)
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


void printMatrices(const std::vector<FpType> A_inv, int matrixSize, int numMatrices)
{
    for (int k = 0; k < numMatrices; ++k)
    {
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                std::cout << A_inv[(i * matrixSize) + (k * matrixSize * matrixSize) + j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}
