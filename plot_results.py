import matplotlib.pyplot as plt
import numpy as np


# Sample data (replace with your actual data)
variable = '10m_u_component_of_wind'
x_labels = ['0.1', '0.5', '1']
ebcc_cr = [4.11, 6.53, 8.28]
lic_tcm_cr = [5.33, 11.01, 17.29]
lic_tcm_cr_from_scratch = [0.62, 1.66, 4.01]
lic_tcm_cr_finetune = [0.71, 2.77, 6.74]

x = np.arange(len(x_labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()

# Plot the bars
bars1 = ax.bar(x - 3*width/2, ebcc_cr, width, label='EBCC')
bars2 = ax.bar(x - width/2, lic_tcm_cr, width, label='LIC_TCM-pretrained')
bars3 = ax.bar(x + width/2, lic_tcm_cr_from_scratch, width, label='LIC_TCM-rand-init')
bars4 = ax.bar(x + 3*width/2, lic_tcm_cr_finetune, width, label='LIC_TCM-finetune')

# Add labels, title, legend, and custom x-axis tick labels
ax.set_xlabel('Error Bound to Uncertainty')
ax.set_ylabel('Compression Ratio')
ax.set_title(f'{variable} Compression Ratio EBCC vs. LIC_TCM')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

# Function to add value labels on top of bars
def add_labels(bars):
  for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Add labels to both sets of bars
add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.tight_layout()
plt.grid(True)
# Save the plot
output_path = f"./results/{variable}_lic_tcm_cr_full_res.png"
plt.savefig(output_path, dpi=500, bbox_inches="tight")
plt.close()
