#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <fstream>
#include <random>

using FpType = float;

int main()
{

    // int matrixSize = matrixSizE;
    // make a vector of value from 1 to 32
    std::vector<int> matrixSizes(32);
    std::iota(matrixSizes.begin(), matrixSizes.end(), 1);

    // for each value in the vector, calculate the adjusted value to the nearest power of 2
    for (int matrixSize : matrixSizes)
    {
        // int matrixSizePow2 = 1 << static_cast<int>(std::ceil(std::log2(matrixSize)));
        int matrixSizePow2 = 1 << (32 - __builtin_clz(matrixSize - 1));

        std::cout << "matrixSize: " << matrixSize << " and adjusted to: " << matrixSizePow2 << '\n';
    }
    // int matrixSizePow2 = 1 << static_cast<int>(std::ceil(std::log2(matrixSize)));
    // int matrixSizePow2 = 1 << (32 - __builtin_clz(matrixSize - 1));

    // std::cout << "matrixSize: " << matrixSize << " and adjusted to: " << matrixSizePow2 << '\n';

    // std::ifstream file("mtrand32_new1.txt");

    // create a matrix with size matrixSizePow2 x matrixSizePow2, but only fill the first matrixSize x matrixSize elements and among the rest fill the diagonal with 1s and the rest with 0s
    // std::vector<FpType> templateMatrix(matrixSizePow2 * matrixSizePow2);
    // for (int i = 0; i < matrixSize; ++i)
    // {
    //     for (int j = 0; j < matrixSize; ++j)
    //     {
    //         file >> templateMatrix[(i * matrixSizePow2) + j];
    //     }
    // }

    // file.close();
    // for (int i = matrixSize; i < matrixSizePow2; ++i)
    // {
    //     for (int j = 0; j < matrixSizePow2; ++j)
    //     {
    //         if (j == i)
    //         {
    //             templateMatrix[(i * matrixSizePow2) + j] = static_cast<FpType>(1.0);
    //         }
    //         else
    //         {
    //             templateMatrix[(i * matrixSizePow2) + j] = static_cast<FpType>(0.0);
    //         }
    //     }
    // }

    // // print the matrix
    // for (int i = 0; i < matrixSizePow2; ++i)
    // {
    //     for (int j = 0; j < matrixSizePow2; ++j)
    //     {
    //         std::cout << templateMatrix[(i * matrixSizePow2) + j] << ' ';
    //     }
    //     std::cout << '\n';
    // }

    return 0;
}
