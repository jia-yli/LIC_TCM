import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def draw_plot(metric_name, metric_unit, train_label):
  '''
  Preparation
  '''
  df = pd.read_csv(f'/users/ljiayong/projects/LIC_TCM/results/eval_tcm_ckpt_era5_{train_label}.csv')

  # variable_lst = ['10m_u_component_of_wind']
  variable_lst =  df['variable'].unique().tolist()

  '''
  Draw
  '''
  for variable in variable_lst:
    plt.figure(figsize=(8, 6))

    '''
    Training ckpt
    '''
    tcm_lambda_lst = sorted(df['lambda'].unique(), reverse=True)
    for tcm_lambda in tcm_lambda_lst:
      model_data = df[(df['variable']==variable) & (df['lambda']==tcm_lambda)].sort_values('epoch')
      plt.plot(
        model_data['epoch'], 
        model_data[metric_name], 
        marker='o',
        markersize=4, 
        label=fr"$\lambda$ = {tcm_lambda}"
      )
    
    plt.title(f'{metric_name} vs. epoch on {variable}')
    plt.xlabel(f'epoch')
    plt.ylabel(f'{metric_name}{metric_unit}')
    plt.legend()
    plt.grid(True)
    output_path = f"./results/train_process_new/train_{train_label}_{variable}_{metric_name}_vs_epoch.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
  # train_label_lst = ["20epoch", "50epoch", "finetune", "full_res", "full_res_1"]
  train_label_lst = ["weighted_finetune"]

  for train_label in train_label_lst:
    metric_names = ["bit_rate", "compression_ratio", "psnr", "ms_ssim"]
    metric_units = ["[bpp]", "", "[dB]", "[dB]"]

    for param in zip(metric_names, metric_units):
      draw_plot(*param, train_label=train_label)



