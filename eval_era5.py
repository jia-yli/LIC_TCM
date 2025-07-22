import os
import time
import h5py
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
from tqdm import tqdm

import pickle
import re

import torch
import torch.nn.functional as F
from models import TCM

class LicTcmCompressor:
  def __init__(self, checkpoint_path1, checkpoint_path2):
    self.device = 'cuda:0'
    match1 = re.search(r'[nN]_(\d+)', checkpoint_path1)
    N1 = int(match1.group(1))
    match2 = re.search(r'[nN]_(\d+)', checkpoint_path2)
    N2 = int(match2.group(1))

    self.net1 = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N1, M=320)
    self.net1 = self.net1.to(self.device)
    self.net1.eval()
    # load check point
    dictory = {}
    checkpoint = torch.load(checkpoint_path1, map_location=self.device, weights_only=True)
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    self.net1.load_state_dict(dictory)
    self.net1.update()

    # net 2
    self.net2 = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N2, M=320)
    self.net2 = self.net2.to(self.device)
    self.net2.eval()
    # load check point
    dictory = {}
    checkpoint = torch.load(checkpoint_path2, map_location=self.device, weights_only=True)
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    self.net2.load_state_dict(dictory)
    self.net2.update()

  @staticmethod
  def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
      x,
      (padding_left, padding_right, padding_top, padding_bottom),
      mode="constant",
      value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

  @staticmethod
  def crop(x, padding):
    return F.pad(
      x,
      (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

  def compress(self, data, net_id, clip_extreme = True):
    '''
    preprocess
    '''
    # Step 1: clip extreme values
    if clip_extreme:
      quantile = 0.2
      range_factor = 2
      Q1 = np.quantile(data, quantile)
      Q3 = np.quantile(data, 1-quantile)
      IQR = Q3 - Q1
      lower_bound = Q1 - range_factor * IQR
      upper_bound = Q3 + range_factor * IQR

      data_extreme_mask = (data < lower_bound) | (data > upper_bound)
      data_extreme_ratio = data_extreme_mask.sum()/data_extreme_mask.size
      data_extreme_positions = np.flatnonzero(data_extreme_mask)
      data_extreme_values = data[data_extreme_mask]

      data_clipped = np.clip(data, lower_bound, upper_bound)
    else:
      data_extreme_positions = np.array([], dtype=np.int64)
      data_extreme_values = np.array([], dtype=np.float32)

      data_clipped = data


    # Step 2: Scale the input
    min_val = data_clipped.min()
    max_val = data_clipped.max()

    if max_val == min_val:
      data_scaled = data_clipped - min_val
    else:
      data_scaled = (data_clipped - min_val) / (max_val - min_val)

    '''
    NN
    '''
    padding_granularity = 128
    num_images = data_scaled.shape[0]
    batch_size = 1
    num_epochs = (num_images + batch_size - 1) // batch_size

    results = []
    for epoch_idx in tqdm(range(num_epochs), total=num_epochs):
      start_idx = epoch_idx * batch_size
      end_idx = min(start_idx + batch_size, num_images)
      data_tensor = torch.Tensor(data_scaled[start_idx:end_idx]).float().to(self.device)
      data_tensor = data_tensor.unsqueeze(1).repeat(1,3,1,1) # [N, C, H, W] with FP32 in Range [0, 1]
      data_tensor, padding = self.pad(data_tensor, padding_granularity)
      with torch.no_grad():
        torch.cuda.synchronize()
        out_enc = eval(f"self.net{net_id}").compress(data_tensor)
        results.append(out_enc)
    
    info = {
      'data_extreme_positions': data_extreme_positions,
      'data_extreme_values': data_extreme_values,
      'min_val': min_val,
      'max_val': max_val,
      'padding': padding
    }

    return results, info

  def decompress(self, results, info, net_id):
    data_extreme_positions = info['data_extreme_positions']
    data_extreme_values = info['data_extreme_values']
    min_val = info['min_val']
    max_val = info['max_val']
    padding = info['padding']

    '''
    NN
    '''
    data_hat_scaled_lst = []
    for out_enc in results:
      with torch.no_grad():
        torch.cuda.synchronize()
        out_dec = eval(f"self.net{net_id}").decompress(out_enc["strings"], out_enc["shape"])
        out_dec["x_hat"] = self.crop(out_dec["x_hat"], padding).mean(dim=-3)
        data_hat_scaled = out_dec["x_hat"].detach().cpu().numpy()
        data_hat_scaled_lst.append(data_hat_scaled)
    data_hat_scaled = np.concatenate(data_hat_scaled_lst, axis=0)
    if max_val == min_val:
      data_hat_clipped = data_hat_scaled + min_val
    else:
      data_hat_clipped = data_hat_scaled * (max_val - min_val) + min_val
    data_extreme_mask = np.zeros(data_hat_clipped.shape, dtype=bool)
    data_extreme_mask.flat[data_extreme_positions] = True
    data_hat_clipped[data_extreme_mask] = data_extreme_values
    
    return data_hat_clipped

  def run_benchmark(self, data, error_bound):
    data, error_bound = data[0:5], error_bound[0:5]
    # import pdb;pdb.set_trace()

    print(f'[INFO] Starting Data Compression......')
    results, info = self.compress(data, 1, clip_extreme=True)
    data_hat = self.decompress(results, info, 1)
    error = np.abs(data - data_hat)
    num_failed_points = (error > error_bound).sum()

    current_info_bytes = (4 + 4) * (len(info['data_extreme_positions']) + 1) + 4 * (1 + 1 + 4) # min, max, padding
    current_data_bytes = 0
    for result in results:
      for s in result['strings']:
        current_data_bytes += len(s[0])
    current_failed_bytes = (4 + 4) * (num_failed_points+1)

    best_compressed_bytes = current_info_bytes + current_data_bytes + current_failed_bytes
    print(f'[INFO] current_info_bytes {current_info_bytes/1e6} MB, current_data_bytes {current_data_bytes/1e6} MB, current_failed_bytes {current_failed_bytes/1e6} MB')
    current_data_hat = data_hat.copy()
    current_compressed_bytes = current_info_bytes + current_data_bytes
    num_residual_runs = 0
    while True:
      print(f'[INFO] Starting Residual Run {num_residual_runs + 1}')
      # get new data_hat using extra residual compression
      residual = data - current_data_hat
      results_residual, info_residual = self.compress(residual, 2, clip_extreme=False)
      residual_hat = self.decompress(results_residual, info_residual, 2)

      current_data_hat = current_data_hat + residual_hat

      data_extreme_positions = info['data_extreme_positions']
      data_extreme_values = info['data_extreme_values']
      data_extreme_mask = np.zeros(current_data_hat.shape, dtype=bool)
      data_extreme_mask.flat[data_extreme_positions] = True
      current_data_hat[data_extreme_mask] = data_extreme_values

      error = np.abs(data - current_data_hat)
      num_failed_points = (error > error_bound).sum()

      current_info_bytes = 4 * (1 + 1 + 4) # min, max, padding
      current_data_bytes = 0
      for result in results_residual:
        for s in result['strings']:
          current_data_bytes += len(s[0])
      current_failed_bytes = (4 + 4) * (num_failed_points+1)

      current_compressed_bytes += current_info_bytes + current_data_bytes
      print(f'[INFO] current_info_bytes {current_info_bytes/1e6} MB, current_data_bytes {current_data_bytes/1e6} MB, current_failed_bytes {current_failed_bytes/1e6} MB')
      print(f'[INFO] current_compressed_bytes {current_compressed_bytes/1e6} MB, best_compressed_bytes {best_compressed_bytes/1e6} MB')
      if (current_compressed_bytes + current_failed_bytes) < best_compressed_bytes:
        best_compressed_bytes = current_compressed_bytes + current_failed_bytes
        num_residual_runs += 1
      else:
        break
    
    print(f'[INFO] Estimated Compressed Size: {best_compressed_bytes/1e6} MB, using {num_residual_runs} residual runs')
    input_size = data.size * data.itemsize
    compression_ratio = input_size/best_compressed_bytes
    failed_bytes_ratio = current_failed_bytes/best_compressed_bytes
    print(f'[INFO] Compression Ratio: {compression_ratio}')
    return best_compressed_bytes, num_residual_runs, num_failed_points, input_size, compression_ratio, failed_bytes_ratio


def compress_hdf5_lic_tcm_pointwise(reanalysis_file, interpolated_ensemble_spread_file, output_hdf5, ebcc_pointwise_max_error_ratio, checkpoint_path1, checkpoint_path2):
  # compression and compression time
  compression_start_time = time.time()

  reanalysis_dataset = xr.open_dataset(reanalysis_file)
  interpolated_ensemble_spread_dataset = xr.open_dataset(interpolated_ensemble_spread_file)
  assert len(reanalysis_dataset.data_vars) == 1
  assert len(interpolated_ensemble_spread_dataset.data_vars) == 1
  assert list(reanalysis_dataset.data_vars)[0] == list(interpolated_ensemble_spread_dataset.data_vars)[0]
  data = reanalysis_dataset[list(reanalysis_dataset.data_vars)[0]].values
  interpolated_ensemble_spread = interpolated_ensemble_spread_dataset[list(interpolated_ensemble_spread_dataset.data_vars)[0]].values
  error_bound = interpolated_ensemble_spread * ebcc_pointwise_max_error_ratio

  compressor = LicTcmCompressor(checkpoint_path1, checkpoint_path2)
  best_compressed_bytes, num_residual_runs, num_failed_points, input_size, compression_ratio, failed_bytes_ratio = compressor.run_benchmark(data, error_bound)

  compression_end_time = time.time()
  compression_time = compression_end_time - compression_start_time

  # input_size = os.path.getsize(input_hdf5)
  # compression_ratio = input_size/best_compressed_bytes
  compression_bandwidth = input_size/1e6/compression_time

  return compression_time, compression_ratio, compression_bandwidth, num_residual_runs, num_failed_points, failed_bytes_ratio

def run_lic_tcm_pointwise(output_path, era5_path, variable, ebcc_pointwise_max_error_ratio, checkpoint_path1, checkpoint_path2):
  if not checkpoint_path2:
    checkpoint_path2 = checkpoint_path1
  reanalysis_file = os.path.join(era5_path, f'single_level/reanalysis/2024/12/{variable}.nc')
  interpolated_ensemble_spread_file = os.path.join(era5_path, f'single_level/interpolated_ensemble_spread/2024/12/{variable}.nc')
  output_hdf5_file_path = os.path.join(output_path, f'{variable}_compressed_lic_tcm_pointwise_ratio_{ebcc_pointwise_max_error_ratio}.hdf5')
  compression_time, compression_ratio, compression_bandwidth, num_residual_runs, num_failed_points, failed_bytes_ratio = compress_hdf5_lic_tcm_pointwise(reanalysis_file, interpolated_ensemble_spread_file, output_hdf5_file_path, ebcc_pointwise_max_error_ratio, checkpoint_path1, checkpoint_path2)
  results = {
    'checkpoint_path1': checkpoint_path1,
    'checkpoint_path2': checkpoint_path2, 
    'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
    'compression_ratio' : compression_ratio,
    'compression_time' : compression_time,
    'compression_bandwidth': compression_bandwidth,
    'num_residual_runs': num_residual_runs,
    'num_failed_points': num_failed_points,
    'failed_bytes_ratio': failed_bytes_ratio,
  }
  return results

if __name__ == '__main__':
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

  # global value
  for variable_idx in range(len(variable_lst)):
    variable = variable_lst[variable_idx]
    era5_path = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5_large'
    output_path = f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/test'
    os.makedirs(output_path, exist_ok = True)

    '''
    Param Combinations
    '''
    ebcc_pointwise_max_error_ratio_lst = [0.1, 0.5, 1]
    ebcc_pointwise_max_error_ratio_lst = [1]
    checkpoint_path1_lst=[
      # '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.05.pth.tar',
      # '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar',
      # '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_era5_20/N_128_lambda_0.05/checkpoint_best.pth.tar',
      # '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_era5_full_res_1/N_128_lambda_0.05/checkpoint_best.pth.tar',
      # '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_era5_full_res_finetune/N_64_lambda_0.05/checkpoint_best.pth.tar',
      f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_finetune/N_64_lambda_0.05_{variable}/checkpoint_best.pth.tar',
      # '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_rand_init/N_64_lambda_0.05/checkpoint_best.pth.tar',
    ]
    checkpoint_path2_lst=['/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar']
    # checkpoint_path2_lst=[None]
    # checkpoint_path1_lst=['/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_fintune/N_64_lambda_0.05/checkpoint_best.pth.tar']
    # checkpoint_path2_lst=[
    #   '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.05.pth.tar',
    #   '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar',
    #   '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_era5_full_res_1/N_128_lambda_0.05/checkpoint_best.pth.tar',
    #   '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_era5_full_res_finetune/N_64_lambda_0.05/checkpoint_best.pth.tar',
    #   '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_fintune/N_64_lambda_0.05/checkpoint_best.pth.tar',
    #   '/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_rand_init/N_64_lambda_0.05/checkpoint_best.pth.tar',
    # ]
    param_combinations = list(itertools.product([output_path], [era5_path], [variable], ebcc_pointwise_max_error_ratio_lst, checkpoint_path1_lst, checkpoint_path2_lst))
    
    '''
    Step 3: Run LIC_TCM with Pointwise Error Bound
    '''
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    # for loop
    results = []
    for params in param_combinations:
      print(f'[INFO] Starting LIC_TCM Pointwise Error Compression with Param: {params}')
      results.append(run_lic_tcm_pointwise(*params))
    
    # Convert results to a structured DataFrame
    results_df = pd.DataFrame(results)

    results_df.to_csv(f'./results/error_bound_pipeline_test/{variable}_sep_lic_tcm_pointwise_compression.csv', index=False)

    # '''
    # Step 4: Plot Compression Error Distribution
    # '''
    # print(f'[INFO] Ploting EBCC for Variable {variable} ......')
    # for params in param_combinations:
    #   plot_compression_error_dist(*params)





