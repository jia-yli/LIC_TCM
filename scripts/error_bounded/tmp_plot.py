import os
import matplotlib.pyplot as plt
import numpy as np
# plot pretrain vs. finetune vs. weighted finetune

def plot_residual_methods(save_path):
  # 10m_u_component_of_wind, single model vs. pretrain res
  data = {
    'Pretrain': [12.89, 12.89],
    'Finetune-MSE': [7.88 , 14.91],
    'Finetune-WeightedMSE': [14.95, 16.70],
  }
  # Sample data (replace with your actual data)
  variable = '10m_u_component_of_wind'
  x_labels = list(data.keys())
  x = np.arange(len(x_labels))  # the label locations
  width = 0.3  # the width of the bars

  fig, ax = plt.subplots()
  # Plot the bars
  bars1 = ax.bar(x - width/2, [data[key][0] for key in x_labels], width, label='Single Model')
  bars2 = ax.bar(x + width/2, [data[key][1] for key in x_labels], width, label='Pretrain Model for Residual')

  # Add labels, title, legend, and custom x-axis tick labels
  ax.set_xlabel('Model')
  ax.set_ylabel('Compression Ratio')
  ax.set_title(f'{variable} Compression Ratio vs. Model')
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

  plt.tight_layout()
  plt.grid(True)
  # Save the plot
  output_path = os.path.join(save_path, f"{variable}_residual_methods.png")
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()


def plot_variables(save_path):
  # pretrained only vs. weighted only vs. weighted + pretrained
  data = {
    "10m_u_component_of_wind": [12.89, 14.96, 16.70],
    "2m_temperature"         : [17.91, 9.66, 21.92],
    "ice_temperature_layer_3": [3.72, 3.55, 3.55],
    "mean_sea_level_pressure": [19.78, 5.08, 25.99],
    "skin_temperature"       : [18.31, 6.02, 23.22],
  }

  # Sample data (replace with your actual data)
  x_labels = list(data.keys())
  x = np.arange(len(x_labels))  # the label locations
  width = 0.25  # the width of the bars

  fig, ax = plt.subplots()
  # Plot the bars
  bars1 = ax.bar(x - width, [data[key][0] for key in x_labels], width, label='Pretrain Only')
  bars2 = ax.bar(x, [data[key][1] for key in x_labels], width, label='Finetune-WeightedMSE Only')
  bars3 = ax.bar(x + width, [data[key][2] for key in x_labels], width, label='Finetune-WeightedMSE + Pretrain for Residual')

  # Add labels, title, legend, and custom x-axis tick labels
  ax.set_xlabel('Variable')
  ax.set_ylabel('Compression Ratio')
  fig.suptitle(f'Compression Ratio vs. Variables')
  ax.set_xticks(x)
  ax.set_xticklabels(x_labels)
  plt.xticks(rotation = 45)
  ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')

  # Function to add value labels on top of bars
  def add_labels(bars):
    for bar in bars:
      height = bar.get_height()
      ax.annotate(f'{height}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  fontsize=7,
                  textcoords="offset points",
                  ha='center', va='bottom')

  # Add labels to both sets of bars
  add_labels(bars1)
  add_labels(bars2)
  add_labels(bars3)

  max_height = max(max(values) for values in data.values())
  ax.set_ylim(0, max_height * 1.2)

  plt.tight_layout()
  plt.grid(True)
  # Save the plot
  output_path = os.path.join(save_path, f"compression_ratio_variables.png")
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()

if __name__ == "__main__":
  save_path = '/users/ljiayong/projects/LIC_TCM/results/summary_20250721'
  plot_residual_methods(save_path)
  plot_variables(save_path)