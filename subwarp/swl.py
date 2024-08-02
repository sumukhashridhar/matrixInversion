import numpy as np


def calc_swl_idx_lei(k, matrixSize, numElements):
    num = np.float32(0.0)
    size_fp32 = num.nbytes
    swl_size = matrixSize * size_fp32
    y_c = (k * size_fp32) // swl_size
    x_c = (k * size_fp32) % swl_size

    x_c_swl = y_c ^ x_c
    x_swz = (x_c_swl * size_fp32 * matrixSize) // size_fp32 % numElements + k % (size_fp32 * matrixSize // size_fp32)

    # print("oldIdx: %d newIdx: %d" % (k, x_swz))

    return x_swz

matrixSize = 8
numElements = matrixSize * matrixSize
a = np.arange(numElements)

# print a as 2D matrix
print("Input matrix:")
print(a.reshape(matrixSize, matrixSize))

swl_idx = [calc_swl_idx_lei(k, matrixSize, numElements) for k in a]

# print swl_idx as 2D matrix
print("\n")
print("SWL idx matrix:")
print(np.array(swl_idx).reshape(matrixSize, matrixSize))
