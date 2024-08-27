def get_2d_index(index, cols):
    row = index // cols
    col = index % cols
    swl_idx = col ^ row

    print("oldIdx: %d newIdx: %d" % (index, swl_idx))

    # return (row, col, swl_idx)
    # return (row, col)

# Example usage
cols = 32
for i in range(cols * 2):
    get_2d_index(i, cols)
    # print(f"1D index {i} -> 2D index {get_2d_index(i, cols)}")
