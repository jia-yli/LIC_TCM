import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Sample data (replace with your actual data)
variable = '10m_u_component_of_wind'
exp_group_name_lst = ['same', '64_pre']
for exp_group_name in exp_group_name_lst:
  csv_path = f'/users/ljiayong/projects/LIC_TCM/results/error_bound_pipeline_full_small/edit_{exp_group_name}_{variable}_lic_tcm_pointwise_compression.csv'
  df = pd.read_csv(csv_path)
  pivot_df = df.pivot(index='Name', columns='ebcc_pointwise_max_error_ratio', values='compression_ratio')

  ax = pivot_df.plot(kind='bar', width=0.8)

  for container in ax.containers:
    ax.bar_label(container, labels=[f'{v:.2f}' for v in container.datavalues], label_type='edge', padding=2)

  # plt.xlabel('Model')
  plt.ylabel('Compression Ratio')
  plt.title(f'{variable} Compression Ratio vs. Model')
  plt.xticks(rotation=22.5)
  plt.legend(title='Error Bound to\nEnsemble Spread', bbox_to_anchor=(1.05, 1), loc='upper left')

  # plt.tight_layout()
  # plt.grid(True)
  # Save the plot
  output_path = f"/users/ljiayong/projects/LIC_TCM/results/error_bound_pipeline_full_small/{exp_group_name}_{variable}.png"
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()
