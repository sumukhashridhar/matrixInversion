import numpy as np

def get_2d_index(index, cols):
    row = index // cols
    col = index % cols

    swl_idx = col ^ row

    if (int(index / cols) > 0):
        swl_idx += cols * (int(index / cols))

    return swl_idx

    # if (index % 7 == 0):
    #     print("oldIdx: %d newIdx: %d" % (index, swl_idx))

    # return (row, col, swl_idx)
    # return (row, col)

# Example usage
cols = 8
rows = 8
orig_lst = []
swl_lst = []
for i in range(cols * rows):
    orig_lst.append(i)
    swl_lst.append(get_2d_index(i, cols))
    # get_2d_index(i, cols)
    # print(f"1D index {i} -> 2D index {get_2d_index(i, cols)}")

# print(swl_lst)

reshaped = np.array(swl_lst).reshape(rows, cols)
# save reshaped to a file
# np.savetxt("reshaped.csv", reshaped, delimiter=",", fmt='%d')
print(np.array(orig_lst).reshape(rows, cols))
print(reshaped)

