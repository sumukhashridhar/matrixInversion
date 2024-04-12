import numpy as np
from scipy.linalg import lu, inv

mtrx = np.array([[4, 11, 3], [4, 10, 4], [2, 4, 2]])
print("Input matrix is:")
print(mtrx)

inv_mtrx = inv(mtrx)
print("Inverse is:")
print(inv_mtrx)

p, l, u = lu(mtrx)

print("U is:")
print(u)
print("L is:")
print(l)