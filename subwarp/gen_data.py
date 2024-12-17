import numpy as np

def generate_test_matrix(size=100, seed=42):
    """
    Generate a well-conditioned square matrix with integer values.
    
    Parameters:
    size (int): Size of the square matrix (default: 100)
    seed (int): Random seed for reproducibility (default: 42)
    
    Returns:
    numpy.ndarray: A square matrix of integers
    """
    np.random.seed(seed)
    
    # Generate a random matrix with integers between -10 and 10
    base_matrix = np.random.randint(-100, 101, size=(size, size))
    
    # Make the matrix diagonally dominant to ensure it's well-conditioned
    # This helps avoid numerical instability in LU factorization
    for i in range(size):
        base_matrix[i][i] = abs(sum(base_matrix[i])) + 10
    
    return base_matrix

def save_matrices_for_testing(max_size=100):
    """
    Generate and save matrices of different sizes for testing LU factorization.
    
    Parameters:
    max_size (int): Maximum size of matrices to generate (default: 32)
    """
    matrices = {}
    for size in range(1, max_size + 1):
        matrices[size] = generate_test_matrix(size)
    
    return matrices

# Generate the full 100x100 matrix
full_matrix = generate_test_matrix(100)

full_matrix.tofile('matrix.txt')

# # Generate matrices for testing (sizes 1 to 32)
# test_matrices = save_matrices_for_testing(32)

# # Example: Print the first 5x5 portion of the 100x100 matrix
# print("First 5x5 portion of the 100x100 matrix:")
# print(full_matrix[:5, :5])

# # Example: Print dimensions of one of the test matrices
# print(f"\nDimensions of test matrix size 32: {test_matrices[32].shape}")