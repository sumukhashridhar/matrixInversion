import numpy as np

# Generate a 64x64 matrix with random values
matrix = np.random.rand(64, 64)

# Save the matrix to csv file
np.savetxt("matrix64.txt", matrix, delimiter=",", dtype=np.float32)

print("64x64 matrix generated and saved to 'matrix64.txt'")