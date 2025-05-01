import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import re


def draw_plot(x_name, x_unit, y_name, y_unit):
  '''
  Preparation
  '''
  # TCM models
  df_tcm = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_tcm_era5.csv')
  # baseline from zoo
  df_zoo = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_zoo_era5.csv')

  variable_lst = df_tcm['variable'].unique()

  '''
  Draw
  '''
  for variable in variable_lst:
    plt.figure(figsize=(8, 6))

    '''
    TCM Model
    '''
    tcm_n_lst = sorted(df_tcm['n'].unique(), reverse=True)
    for tcm_n in tcm_n_lst:
      model_data = df_tcm[(df_tcm['variable']==variable) & (df_tcm['n']==tcm_n)].sort_values('lambda')
      plt.plot(
        model_data[x_name], 
        model_data[y_name], 
        marker='o',
        markersize=4, 
        label=f"LIC_TCM-{tcm_n}(CVPR23)"
      )
    
    # '''
    # Zoo Models
    # '''
    # model_name_lst = df_zoo['model_name'].unique()
    # for model_name in model_name_lst:
    #   model_data = df_zoo[(df_zoo['dataset_name']==dataset_name) & (df_zoo['model_name']==model_name)].sort_values('quality_factor')
    #   plt.plot(
    #     model_data[x_name], 
    #     model_data[y_name], 
    #     marker='o',
    #     markersize=4, 
    #     label=f"{model_name}"
    #   )

    '''
    Metric Target Values
    '''
    pattern = re.compile(rf"^{re.escape(y_name)}_goal_(.+)$")
    error_bound_to_uncertainty_lst = []
    for col in df_tcm[(df_tcm['variable']==variable)].columns:
      match = pattern.match(col)
      if match:
        error_bound_to_uncertainty_lst.append(match.group(1))
    
    # Draw a horizontal line for each alpha
    ax = plt.gca()
    for i, error_bound_to_uncertainty in enumerate(error_bound_to_uncertainty_lst):
      target_value = df_tcm[(df_tcm['variable']==variable)][f"{y_name}_goal_{error_bound_to_uncertainty}"].iloc[0]
      ax.axhline(y=target_value, color='red', linestyle='dotted', linewidth=1)
      # Label on the top-left
      ax.text(ax.get_xlim()[0], target_value, f" Error = {error_bound_to_uncertainty}x Uncertainty", va='top', ha='left', fontsize=8, color='red')
      ax.text(ax.get_xlim()[0], target_value, f" {target_value:.2f}{y_unit}", va='bottom', ha='left', fontsize=8, color='red')
    
    plt.title(f'{y_name} vs. {x_name} on {variable}')
    plt.xlabel(f'{x_name}{x_unit}')
    plt.ylabel(f'{y_name}{y_unit}')
    plt.legend()
    plt.grid(True)
    output_path = f"./results/eval_{variable}_{y_name}_vs_{x_name}.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
  x_names = ["bit_rate", "compression_ratio"]
  x_units = ["[bpp]", ""]
  y_names = ["psnr", "ms_ssim"]
  y_units = ["[dB]", "[dB]"]

  params = [x + y for x, y in itertools.product(zip(x_names, x_units), zip(y_names, y_units))]

  for param in params:
    draw_plot(*param)



