# matrixInversion

## Contents for my Masters thesis "Optimized Block-Level Matrix Inversion Kernels for Small, Batched Matrices on GPUs"

### Abstract

This thesis examines efficient matrix inversion techniques on NVIDIA GPUs, focusing on two primary methods: LU decomposition (LUD) for square matrices and singular value decomposition (SVD) for non-square matrices. We confine ourselves to small/medium sized matrices, execute them in batches of order 1,00,000. We do this efficiently by doing the inversion of a matrix on a single thread block. Finally, we compare and analyze these methods, and benchmark them separately in cuBLAS. Our objective is to develop an efficient implementation of batched matrix inversion operations for integration into the SeisSol framework.