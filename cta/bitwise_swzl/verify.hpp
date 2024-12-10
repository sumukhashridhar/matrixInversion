#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <fstream>

using FpType = float;

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

void verifyLU(const std::vector<FpType> A, std::vector<FpType> LU, int matrixSize, int numMatrices)
{
    // store L and U matrices separately from LU
    std::vector<FpType> L(matrixSize * matrixSize * numMatrices, 0.0);
    std::vector<FpType> U(matrixSize * matrixSize * numMatrices, 0.0);

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
    FpType diff = static_cast<FpType>(0.0);
    int corrLU = 0, wrngLU = 0, offset = 0;

#pragma omp parallel for collapse(1)
    for (int k = 0; k < numMatrices; k++)
    {
        offset = k * matrixSize * matrixSize;
        diff = static_cast<FpType>(0.0);

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
                diff += std::fabs(A[(i * matrixSize) + offset + j] - result);
            }
        }

        if (fabs(diff) < 1e-3)
        {
            corrLU++;
        }
        else
        {
            wrngLU++;
            // printf("Difference: %f\n", diff);
        }
    }

    std::cout << "Correct LU decompositions: " << corrLU << '\n';
    std::cout << "Incorrect LU decompositions: " << wrngLU << '\n';

    // printf("\n");
    // printMatrices(LU, matrixSize, numMatrices);
}
