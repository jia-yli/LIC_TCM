import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def draw_plot(x_name, x_unit, y_name, y_unit):
  '''
  Preparation
  '''
  df_pretrained = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_tcm.csv')
  df_zoo = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_zoo.csv')
  df_custom_300000 = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_tcm_ckpt_300000.csv')
  df_custom_50000 = pd.read_csv('/users/ljiayong/projects/LIC_TCM/results/eval_tcm_ckpt.csv')

  dataset_lst = ['Kodak']

  '''
  Draw
  '''
  for dataset_name in dataset_lst:
    plt.figure(figsize=(8, 6))

    '''
    TCM pretrained ckpt
    '''
    tcm_n_lst = sorted(df_pretrained['n'].unique(), reverse=True)
    for tcm_n in tcm_n_lst:
      model_data = df_pretrained[(df_pretrained['dataset_name']==dataset_name) & (df_pretrained['n']==tcm_n)].sort_values('lambda')
      plt.plot(
        model_data[x_name], 
        model_data[y_name], 
        marker='o',
        markersize=4, 
        label=f"LIC_TCM-{tcm_n} Pretrained"
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

    '''
    Custom ckpt on 300k images
    '''
    tcm_n_lst = sorted(df_custom_300000['n'].unique(), reverse=True)
    for tcm_n in tcm_n_lst:
      model_data = df_custom_300000[(df_custom_300000['dataset_name']==dataset_name) & (df_custom_300000['n']==tcm_n) & (df_custom_300000['label']=='best')].sort_values('lambda')
      plt.plot(
        model_data[x_name], 
        model_data[y_name], 
        marker='x',
        linestyle='--',
        markersize=4, 
        label=f"LIC_TCM-{tcm_n} 300k ImageNet"
      )
    
    '''
    Custom ckpt on 50k images
    '''
    tcm_n_lst = sorted(df_custom_50000['n'].unique(), reverse=True)
    for tcm_n in tcm_n_lst:
      model_data = df_custom_50000[(df_custom_50000['dataset_name']==dataset_name) & (df_custom_50000['n']==tcm_n) & (df_custom_50000['label']=='best')].sort_values('lambda')
      plt.plot(
        model_data[x_name], 
        model_data[y_name], 
        marker='x',
        linestyle='--',
        markersize=4, 
        label=f"LIC_TCM-{tcm_n} 50k ImageNet"
      )
    
    plt.title(f'{y_name} vs. {x_name} on {dataset_name} dataset')
    plt.xlabel(f'{x_name}{x_unit}')
    plt.ylabel(f'{y_name}{y_unit}')
    plt.legend()
    plt.grid(True)
    output_path = f"./results/train_{dataset_name}_{y_name}_vs_{x_name}.png"
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



