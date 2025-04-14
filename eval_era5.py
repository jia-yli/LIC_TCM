import os
import time
import h5py
import xarray as xr
import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
from scipy.interpolate import griddata
from tqdm import tqdm

import pickle

import torch
import torch.nn.functional as F
from models import TCM

def convert_nc_to_hdf5(nc_file, hdf5_file):
  """
  Convert a NetCDF (.nc) file to HDF5 (.h5) format.

  Parameters:
    nc_file (str): Path to the input NetCDF file.
    hdf5_file (str): Path to the output HDF5 file.
  """
  # Open the NetCDF file
  dataset = xr.open_dataset(nc_file)

  # Create an HDF5 file
  with h5py.File(hdf5_file, 'w') as hdf5_f:
    for var_name, da in dataset.data_vars.items():
      data = da.values[0:] # Convert xarray DataArray to NumPy array
      hdf5_f.create_dataset(var_name, data=data)

def spatial_interpolation(data, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, start_idx=0, end_idx=None):
  data = data[start_idx:end_idx]
  num_time_steps = data.shape[0]
  data_interpolated = np.empty((num_time_steps, lon_target_grid.shape[0], lon_target_grid.shape[1]))
  points = np.column_stack((lat_source_grid.ravel(), lon_source_grid.ravel()))
  for t_idx in range(num_time_steps):
    values = data[t_idx].ravel()
    data_interpolated[t_idx] = griddata(points, values, (lat_target_grid, lon_target_grid), method='linear')
  return data_interpolated

def interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_file, output_file):
  # Load reanalysis and ensemble datasets
  ds_reanalysis = xr.open_dataset(reanalysis_file)
  ds_ensemble = xr.open_dataset(ensemble_file)

  # Extract coordinates from reanalysis dataset (target grid)
  time_target = ds_reanalysis['valid_time'].values
  lat_target = ds_reanalysis['latitude'].values
  lon_target = ds_reanalysis['longitude'].values

  # Extract coordinates and spread variable from ensemble dataset (source grid)
  time_source = ds_ensemble['valid_time'].values
  lat_source = ds_ensemble['latitude'].values
  lon_source = ds_ensemble['longitude'].values

  # shape: (Time, Latitude, Longitude)
  assert len(list(ds_ensemble.data_vars)) == 1
  for var_name, da in ds_ensemble.data_vars.items():
    data_source = da.values  # Convert xarray DataArray to NumPy array
    # Step 1: Interpolate Spatial Dims
    # handle longitude wrap-up at 360
    # Src
    lon_source_extended = np.concatenate((lon_source, lon_source[0:1] + 360), axis=0)
    lat_source_grid, lon_source_grid = np.meshgrid(lat_source, lon_source_extended, indexing='ij')
    data_extended = np.concatenate((data_source, data_source[:, :, 0:1]), axis=2)
    # Dst
    lat_target_grid, lon_target_grid = np.meshgrid(lat_target, lon_target, indexing='ij')

    num_time_steps = 8
    num_jobs = (data_extended.shape[0] + num_time_steps - 1) // num_time_steps

    # mp
    with mp.Pool(processes=32) as pool:  # Adjust processes as needed
      results = [pool.apply_async(spatial_interpolation,
        (data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0]))
      ) for idx in range(num_jobs)]
      results = [result.get() for result in results]
    
    # for loop
    # results = [spatial_interpolation(
    #   data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0])
    # ) for idx in range(num_jobs)]

    results = np.concatenate(results, axis=0)

    # Step 2: Interpolate Temporal Dim
    ds_interp_space = xr.Dataset(
      {
        var_name: (['valid_time', 'latitude', 'longitude'], results)
      },
      coords={
        'valid_time': time_source,
        'latitude': lat_target,
        'longitude': lon_target
      }
    )
    # Interpolate in time to match reanalysis time grid
    ds_interp_time = ds_interp_space.interp(
      valid_time=time_target, 
      method="linear")
    ds_output = ds_interp_time.ffill(dim="valid_time")

    # Save to new hdf5 file
    with h5py.File(output_file, 'w') as hdf5_f:
      for var_name, da in ds_output.data_vars.items():
        data = da.values.astype(np.float32)[0:] # Convert xarray DataArray to NumPy array
        hdf5_f.create_dataset(var_name, data=data)

class LicTcmCompressor:
  def __init__(self, checkpoint_path):
    self.device = 'cuda:0'
    self.net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    self.net = self.net.to(self.device)
    self.net.eval()
    # load check point
    dictory = {}
    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    self.net.load_state_dict(dictory)
    self.net.update()

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

  def compress(self, data, clip_extreme = True):
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
        out_enc = self.net.compress(data_tensor)
        results.append(out_enc)
    
    info = {
      'data_extreme_positions': data_extreme_positions,
      'data_extreme_values': data_extreme_values,
      'min_val': min_val,
      'max_val': max_val,
      'padding': padding
    }

    return results, info

  def decompress(self, results, info):
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
        out_dec = self.net.decompress(out_enc["strings"], out_enc["shape"])
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
    # data, error_bound = data[0:5], error_bound[0:5]

    print(f'[INFO] Starting Data Compression......')
    results, info = self.compress(data, clip_extreme=True)
    data_hat = self.decompress(results, info)
    error = np.abs(data - data_hat)
    num_failed_points = (error > error_bound).sum()

    current_info_bytes = (4 + 4) * len(info['data_extreme_positions'] + 1) + 4 * (1 + 1 + 4) # min, max, padding
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
      results_residual, info_residual = self.compress(residual, clip_extreme=False)
      residual_hat = self.decompress(results_residual, info_residual)

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

    return best_compressed_bytes


def compress_hdf5_lic_tcm_pointwise(input_hdf5, input_uncertainty_hdf5, output_hdf5, ebcc_pointwise_max_error_ratio, checkpoint_path):
  # compression and compression time
  compression_start_time = time.time()
  
  with h5py.File(input_hdf5, 'r') as hdf5_in:
    with h5py.File(input_uncertainty_hdf5, 'r') as hdf5_uncertainty_in:
      with h5py.File(output_hdf5, 'w') as hdf5_out:
        assert len(list(hdf5_in.keys())) == 1
        var_name = list(hdf5_in.keys())[0]
        data = np.array(hdf5_in[var_name])  # Read dataset, 1 month data: (744, 721, 1440)
        error_bound = np.array(hdf5_uncertainty_in[var_name]) * ebcc_pointwise_max_error_ratio
        compressor = LicTcmCompressor(checkpoint_path)
        best_compressed_bytes = compressor.run_benchmark(data, error_bound)

  compression_end_time = time.time()
  compression_time = compression_end_time - compression_start_time

  input_size = os.path.getsize(input_hdf5)
  compression_ratio = input_size/best_compressed_bytes
  compression_bandwidth = input_size/1e6/compression_time

  return compression_time, compression_ratio, compression_bandwidth

def run_lic_tcm_pointwise(output_path, variable, ebcc_pointwise_max_error_ratio, checkpoint_path):
  input_hdf5_file_path = os.path.join(output_path, f'{variable}.hdf5')
  input_uncertainty_file_path = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
  output_hdf5_file_path = os.path.join(output_path, f'{variable}_compressed_lic_tcm_pointwise_ratio_{ebcc_pointwise_max_error_ratio}.hdf5')
  compression_time, compression_ratio, compression_bandwidth = compress_hdf5_lic_tcm_pointwise(input_hdf5_file_path, input_uncertainty_file_path, output_hdf5_file_path, ebcc_pointwise_max_error_ratio, checkpoint_path)
  results = {
    'ebcc_pointwise_max_error_ratio' : ebcc_pointwise_max_error_ratio, 
    'compression_ratio' : compression_ratio,
    'compression_time' : compression_time,
    'compression_bandwidth': compression_bandwidth,
  }
  import pdb;pdb.set_trace()
  return results

if __name__ == '__main__':
  variable_lst = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation"
  ]

  # global value
  for variable_idx in range(len(variable_lst)):
    variable = variable_lst[variable_idx]
    output_path = f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/test'
    os.makedirs(output_path, exist_ok = True)

    # '''
    # Step 1: NetCDF to HDF5 without compression
    # '''
    # print(f'[INFO] Converting NetCDF to HDF5 for Variable {variable} ......')
    # nc_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    # hdf5_file = os.path.join(output_path, f'{variable}.hdf5')
    # convert_nc_to_hdf5(nc_file, hdf5_file)


    # '''
    # Step 2: Interpolate Ensemble Spread
    # '''
    # print(f'[INFO] Interpolating Ensemble Spread for Variable {variable} ......')
    # reanalysis_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/reanalysis/{variable}.nc'
    # ensemble_file = f'/capstor/scratch/cscs/ljiayong/datasets/ERA5/ensemble_spread/{variable}.nc'
    # output_file = os.path.join(output_path, f'{variable}_interpolated_ensemble_spread.hdf5')
    # interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_file, output_file)

    '''
    Param Combinations
    '''
    # ebcc_pointwise_max_error_ratio_lst = [0.1, 0.5, 1]
    ebcc_pointwise_max_error_ratio_lst = [0.1]
    checkpoint_path='/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar'

    param_combinations = list(itertools.product([output_path], [variable], ebcc_pointwise_max_error_ratio_lst, [checkpoint_path]))
    
    '''
    Step 3: Run LIC_TCM with Pointwise Error Bound
    '''
    # for loop
    results = []
    for params in param_combinations:
      print(f'[INFO] Starting LIC_TCM Pointwise Error Compression with Param: {params}')
      results.append(run_lic_tcm_pointwise(*params))
    
    # Convert results to a structured DataFrame
    # results_df = pd.DataFrame(results)

    # results_df.to_csv(f'./results/{variable}_ebcc_pointwise_compression.csv', index=False)

    # '''
    # Step 4: Plot Compression Error Distribution
    # '''
    # print(f'[INFO] Ploting EBCC for Variable {variable} ......')
    # for params in param_combinations:
    #   plot_compression_error_dist(*params)





