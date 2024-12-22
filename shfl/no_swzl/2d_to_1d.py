import numpy as np

def get_2d_index(index, cols):
    row = index // cols
    col = index % cols

    swl_idx = col ^ row

    # if (int(index / cols) > 0):
    #     swl_idx += cols * (int(index / cols))

    return swl_idx

    # if (index % 7 == 0):
    #     print("oldIdx: %d newIdx: %d" % (index, swl_idx))

    # return (row, col, swl_idx)
    # return (row, col)

# Example usage
cols = 32
rows = 16
lst = []
for i in range(cols * rows):
    lst.append(get_2d_index(i, cols))
    # get_2d_index(i, cols)
    # print(f"1D index {i} -> 2D index {get_2d_index(i, cols)}")

# print(lst)

reshaped = np.array(lst).reshape(rows, cols)
# save reshaped to a file
np.savetxt("reshaped.csv", reshaped, delimiter=",", fmt='%d')
# print(reshaped)

