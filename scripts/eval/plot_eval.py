import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def draw_plot(x_name, x_unit, y_name, y_unit):
  '''
  Preparation
  '''
  # TCM models
  df_tcm = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_tcm_kodak.csv')
  # baseline from zoo
  df_zoo = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_zoo_kodak.csv')

  # dataset_lst = ['Kodak', 'CLIC']
  dataset_lst = ['Kodak']

  '''
  Draw
  '''
  for dataset_name in dataset_lst:
    plt.figure(figsize=(8, 6))

    '''
    TCM Model
    '''
    tcm_n_lst = sorted(df_tcm['n'].unique(), reverse=True)
    for tcm_n in tcm_n_lst:
      model_data = df_tcm[(df_tcm['dataset_name']==dataset_name) & (df_tcm['n']==tcm_n)].sort_values('lambda')
      plt.plot(
        model_data[x_name], 
        model_data[y_name], 
        marker='o',
        markersize=4, 
        label=f"LIC_TCM-{tcm_n}(CVPR23)"
      )
    
    '''
    Zoo Models
    '''
    model_name_lst = df_zoo['model_name'].unique()
    for model_name in model_name_lst:
      model_data = df_zoo[(df_zoo['dataset_name']==dataset_name) & (df_zoo['model_name']==model_name)].sort_values('quality_factor')
      plt.plot(
        model_data[x_name], 
        model_data[y_name], 
        marker='o',
        markersize=4, 
        label=f"{model_name}"
      )
    
    plt.title(f'{y_name} vs. {x_name} on {dataset_name} dataset')
    plt.xlabel(f'{x_name}{x_unit}')
    plt.ylabel(f'{y_name}{y_unit}')
    plt.legend()
    plt.grid(True)
    output_path = f"./results/eval_{dataset_name}_{y_name}_vs_{x_name}.png"
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



