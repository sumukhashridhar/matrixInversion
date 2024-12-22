import numpy as np

def swizzling_index_2D_to_1D(rows, cols):
    # create an array from 0 to cols
    # then tile it to create a 2D array
    # this 2D array will be used to calculate the swizz
    a = np.arange(cols)
    # shm = np.tile(a, rows)
    # shm = shm.reshape(rows, cols)
    shm = np.zeros((rows, cols), dtype=int)
    # print(shm)
    
    count = -1
    for y in range(rows):
        count += 1
        for x in range(cols):
            # if x % cols == cols - 1:
            #     y += 1
            #     print("y: ", y)
            swizzled_x = x ^ y
            shm[y, swizzled_x % cols] = y * cols + x
    
    return shm

def linearize_2D_array(array):
    return array.flatten()

rows = 4
cols = 32

swizzled_shm_2D = swizzling_index_2D_to_1D(rows, cols)
linearized_swizzled_shm = linearize_2D_array(swizzled_shm_2D)

print("2D Shared Memory Layout with Swizzling:")
print(swizzled_shm_2D)

# # To visualize the indices more clearly
# print("\n2D Shared Memory Layout with Swizzling (Formatted):")
# for row in swizzled_shm_2D:
#     print(" ".join(f"{val:2}" for val in row))

# print("\nLinearized Shared Memory Layout with Swizzling:")
# print(linearized_swizzled_shm)

# # To visualize the indices more clearly in the linearized form
# print("\nLinearized Shared Memory Layout with Swizzling (Formatted):")
# print(" ".join(f"{val:2}" for val in linearized_swizzled_shm))
