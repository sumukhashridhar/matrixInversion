import json
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the JSON file
with open("benchmark_results_1M.json", 'r') as f:
    data = json.load(f)


# Extract data
matrix_sizes = []
runtime_avgs = []
variances = []
std_devs = []
incorrect_inversions = []
regs = []


with open("nohup_swzl_subwarp.out", 'r') as reg_f:
    for line in reg_f:
        if "registers" in line:
            match = re.search(r"Used (\d+) registers", line)
            if match:
                num_regs = int(match.group(1))
                regs.append(num_regs)
                # print(f"Used {match.group(1)} registers")

# Sort the data by matrix size
for key in sorted(data.keys(), key=lambda x: data[x]['matrix_size']):
    matrix_sizes.append(data[key]['matrix_size'])
    runtime_avgs.append(data[key]['runtime_avg'])
    variances.append(data[key]['variance'])
    std_devs.append(data[key]['std_dev'])
    incorrect_inversions.append(np.mean(data[key]['incorrect_inversions']))

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=500)
fig.suptitle('Swizzle Performance Analysis Metrics', fontsize=16, y=0.95)

# 1. Average Runtime Plot
ax1.plot(matrix_sizes, runtime_avgs, 'b-o', linewidth=2, markersize=6)
ax1.set_title('Average Runtime vs Matrix Size')
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Average Runtime (milli seconds)')
ax1.grid(True, linestyle='--', alpha=0.7)

# 2. Variance Plot
ax2.plot(matrix_sizes, variances, 'r-o', linewidth=2, markersize=6)
ax2.set_title('Runtime Variance vs Matrix Size')
ax2.set_xlabel('Matrix Size')
ax2.set_ylabel('Variance')
ax2.grid(True, linestyle='--', alpha=0.7)
# Use scientific notation for small values
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 3. Standard Deviation Plot
ax3.plot(matrix_sizes, std_devs, 'g-o', linewidth=2, markersize=6)
ax3.set_title('Standard Deviation vs Matrix Size')
ax3.set_xlabel('Matrix Size')
ax3.set_ylabel('Standard Deviation')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# # 4. Registers Plot
# ax4.plot(matrix_sizes, regs, 'm-o', linewidth=2, markersize=6)
# ax4.set_title('Registers vs Matrix Size')
# ax4.set_xlabel('Matrix Size')
# ax4.set_ylabel('Registers Used', color='m', fontsize=12, fontweight='bold')
# ax4.set_ylim(0, np.max(regs) + 10)
# ax4.grid(True, linestyle='--', alpha=0.7)
# ax4.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 4. Incorrect Inversions Plot
ax4.plot(matrix_sizes, incorrect_inversions, 'm-o', linewidth=2, markersize=6)
ax4.set_title('Incorrect Inversions vs Matrix Size')
ax4.set_xlabel('Matrix Size')
ax4.set_ylabel('Count of Incorrect Inversions')
ax4.grid(True, linestyle='--', alpha=0.7)

# If all incorrect_inversions are 0, set y-axis limit to emphasize this
if all(x == 0 for x in incorrect_inversions):
    ax4.set_ylim(-0.1, 1.1)
    ax4.text(np.mean(matrix_sizes), 0.5, 'All inversions correct (0 errors)', 
             horizontalalignment='center', verticalalignment='center')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('swzl_performance_analysis.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average Runtime Range: {min(runtime_avgs):.6f} to {max(runtime_avgs):.6f} seconds")
print(f"Maximum Variance: {max(variances):.6e}")
print(f"Maximum Standard Deviation: {max(std_devs):.6e}")
print(f"Total Incorrect Inversions: {sum(incorrect_inversions)}")
