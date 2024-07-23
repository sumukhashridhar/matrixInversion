def generalized_map_number(left_number, multiplier):
    modulus = multiplier * multiplier
    base = (left_number * multiplier) & (modulus - 1)
    remainder = base & (multiplier - 1)
    
    # Adjust the base value based on the remainder
    return base + remainder

# Function to create the mapping for numbers 0 to (modulus - 1)
def create_mapping(multiplier):
    modulus = multiplier * multiplier
    mapping = {i: generalized_map_number(i, multiplier) for i in range(modulus)}
    return mapping

# Define the multiplier
multiplier = 4

# Generate the mapping
mapping = create_mapping(multiplier)

# Print the mapping
for key, value in mapping.items():
    print(f"{key} -> {value}")
