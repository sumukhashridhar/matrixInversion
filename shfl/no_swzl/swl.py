import numpy as np


def calc_swl_idx_lei(k, matrixSize, numElements):
    size_fp32 = np.float32(0.0).nbytes
    swl_size = numElements * size_fp32
    y_c = (k * size_fp32) // swl_size
    x_c = (k * size_fp32) % swl_size

    x_c_swl = y_c ^ x_c
    x_swz = (x_c_swl * size_fp32 * matrixSize) // size_fp32 % numElements + k % (size_fp32 * matrixSize // size_fp32)

    # print("oldIdx: %d newIdx: %d" % (k, x_swz))

    return x_swz

matrixSize = 32
numElements = matrixSize * 2
a = np.arange(matrixSize)
# make a 2D matrix of size matrixSize x matrixSize
a = np.tile(a, 2)

# print a as 2D matrix
print("Input matrix:")
print(a.reshape(2, matrixSize))

swl_idx = [calc_swl_idx_lei(k, matrixSize, numElements) for k in a.reshape(2, matrixSize)]

# print swl_idx as 2D matrix
print("\n")
print("SWL idx matrix:")
print(np.array(swl_idx).reshape(2, matrixSize))
