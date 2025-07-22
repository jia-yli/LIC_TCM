import pandas as pd
import matplotlib.pyplot as plt
import os

variable_lst = [
  # "100m_u_component_of_wind",
  # "100m_v_component_of_wind",
  "10m_u_component_of_wind",
  # "10m_v_component_of_wind",
  # "2m_dewpoint_temperature",
  "2m_temperature",
  # "ice_temperature_layer_1",
  # "ice_temperature_layer_2",
  "ice_temperature_layer_3",
  # "ice_temperature_layer_4",
  # "maximum_2m_temperature_since_previous_post_processing",
  "mean_sea_level_pressure",
  # "minimum_2m_temperature_since_previous_post_processing",
  # "sea_surface_temperature",
  "skin_temperature",
  # "surface_pressure",
  # "total_precipitation",
]

def plot(variable_lst, col_name):
  variable_values = {}
  for variable in variable_lst:
    df = pd.read_csv(f'./results/error_bound_pipeline_test/{variable}_lic_tcm_pointwise_compression.csv')
    value = df[col_name].values[0]
    variable_values[variable] = value

  # Plotting
  plt.figure(figsize=(10, 6))
  bars = plt.bar(variable_values.keys(), variable_values.values())

  for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
              f'{height:.2f}', ha='center', va='bottom')

  plt.xlabel("Variable")
  plt.ylabel(f"{col_name}")
  plt.title(f"{col_name} for Bound = 1x Ensemble Spread")
  plt.xticks(rotation=90)
  plt.grid(True)
  output_path = f"/users/ljiayong/projects/LIC_TCM/results/error_bound_pipeline_test/{col_name}.png"
  plt.savefig(output_path, dpi=500, bbox_inches="tight")
  plt.close()

if __name__ == '__main__':
  plot(variable_lst, 'compression_ratio')
  plot(variable_lst, 'failed_bytes_ratio')
  plot(variable_lst, 'num_residual_runs')
