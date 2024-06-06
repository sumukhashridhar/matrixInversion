#include <iostream>
#include <vector>
#include <cmath>

using FpType = float;

void verifyLU(const std::vector<FpType> &A, const std::vector<FpType> &L, const std::vector<FpType> &U, int matrixSize, int numMatrices)
{
    int wrngLU = 0, corrLU = 0, nanLU = 0;

#pragma omp parallel for reduction(+ : wrngLU, corrLU, nanLU) collapse(1)
    for (int k = 0; k < numMatrices; ++k)
    {
        int offset = k * matrixSize * matrixSize;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                FpType sum = 0.0;
                for (int l = 0; l < matrixSize; ++l)
                {
                    sum += L[i * matrixSize + offset + l] * U[l * matrixSize + offset + j];
                }
                FpType diff = std::fabs(sum - A[i * matrixSize + offset + j]);
                if (std::isnan(diff))
                {
#pragma omp atomic
                    nanLU++;
                }
                else if (diff < 1e-6)
                {
#pragma omp atomic
                    corrLU++;
                }
                else
                {
#pragma omp atomic
                    wrngLU++;
                }
            }
        }
    }

    std::cout << "Correct LU decompositions: " << corrLU << '\n';
    std::cout << "Incorrect LU decompositions: " << wrngLU << '\n';
    std::cout << "NaN LU decompositions: " << nanLU << '\n';
}

void verifyInv(const std::vector<FpType> &A, const std::vector<FpType> &A_inv, int matrixSize, int numMatrices)
{
    int corrInv = 0, wrngInv = 0;

#pragma omp parallel for reduction(+ : corrInv, wrngInv) collapse(1)
    for (int k = 0; k < numMatrices; ++k)
    {
        int offset = k * matrixSize * matrixSize;
        int idCnt = 0;
        for (int i = 0; i < matrixSize; ++i)
        {
            for (int j = 0; j < matrixSize; ++j)
            {
                FpType result = 0.0;
                for (int l = 0; l < matrixSize; ++l)
                {
                    result += A[i * matrixSize + offset + l] * A_inv[j * matrixSize + offset + l];
                }
                if (i == j && std::fabs(result - 1.0) < 1e-6)
                {
                    idCnt++;
                }
            }
        }
        if (idCnt == matrixSize)
        {
#pragma omp atomic
            corrInv++;
        }
        else
        {
#pragma omp atomic
            wrngInv++;
        }
    }

    std::cout << "Correct inversions: " << corrInv << '\n';
    std::cout << "Incorrect inversions: " << wrngInv << '\n';
}
