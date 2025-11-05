import os
import argparse

import xarray as xr
import numpy as np
import multiprocessing as mp

from scipy.interpolate import griddata
from tqdm import tqdm

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
  # shape: (Time, Pressure, Latitude, Longitude)
  assert len(list(ds_ensemble.data_vars)) == 1
  for var_name, da in ds_ensemble.data_vars.items():
    data_source = da.values  # Convert xarray DataArray to NumPy array
    # Step 1: Interpolate Spatial Dims
    # handle longitude wrap-up at 360
    # Src
    lon_source_extended = np.concatenate((lon_source, lon_source[0:1] + 360), axis=0)
    lat_source_grid, lon_source_grid = np.meshgrid(lat_source, lon_source_extended, indexing='ij')
    data_extended = np.concatenate((data_source, data_source[..., 0:1]), axis=-1)
    extended_shape = list(data_extended.shape)
    data_extended = data_extended.reshape(-1, extended_shape[-2], extended_shape[-1])
    # Dst
    lat_target_grid, lon_target_grid = np.meshgrid(lat_target, lon_target, indexing='ij')

    num_time_steps = 8
    num_jobs = (data_extended.shape[0] + num_time_steps - 1) // num_time_steps

    # mp
    with mp.Pool(processes=16) as pool:  # Adjust processes as needed
      results = [pool.apply_async(spatial_interpolation,
        (data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0]))
      ) for idx in range(num_jobs)]
      results = [result.get() for result in results]
    
    # for loop
    # results = [spatial_interpolation(
    #   data_extended, lat_source_grid, lon_source_grid, lat_target_grid, lon_target_grid, idx*num_time_steps, min((idx+1)*num_time_steps, data_extended.shape[0])
    # ) for idx in range(num_jobs)]
    extended_shape[-2] = len(lat_target)
    extended_shape[-1] = len(lon_target)
    results = np.concatenate(results, axis=0).reshape(extended_shape)

    # Step 2: Interpolate Temporal Dim
    if len(extended_shape) == 3:
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
    elif len(extended_shape) == 4:
      ds_interp_space = xr.Dataset(
        {
          var_name: (['valid_time', 'pressure_level', 'latitude', 'longitude'], results)
        },
        coords={
          'valid_time': time_source,
          'pressure_level': ds_reanalysis['pressure_level'].values,
          'latitude': lat_target,
          'longitude': lon_target
        }
      )
    else:
      raise
    # Interpolate in time to match reanalysis time grid
    ds_interp_time = ds_interp_space.interp(
      valid_time=time_target, 
      method="linear")
    ds_output = ds_interp_time.ffill(dim="valid_time")
    ds_output = ds_output.astype(data_source.dtype)

    ds_output.to_netcdf(output_file)

    return True


def main():
  # argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--variable-lst", 
    nargs="+",
    type=str,
    required=True,
    help="List of variables"
  )
  args = parser.parse_args()

  variable_lst = args.variable_lst
  # variable_lst = [
  #   "100m_u_component_of_wind",
  #   "100m_v_component_of_wind",
  #   "10m_u_component_of_wind",
  #   "10m_v_component_of_wind",
  #   "2m_dewpoint_temperature",
  #   "2m_temperature",
  #   "ice_temperature_layer_1",
  #   "ice_temperature_layer_2",
  #   "ice_temperature_layer_3",
  #   "ice_temperature_layer_4",
  #   "maximum_2m_temperature_since_previous_post_processing",
  #   "mean_sea_level_pressure",
  #   "minimum_2m_temperature_since_previous_post_processing",
  #   "sea_surface_temperature",
  #   "skin_temperature",
  #   "surface_pressure",
  #   "total_precipitation",
  # ]

  era5_root = "/capstor/scratch/cscs/ljiayong/datasets/ERA5_large"
  # year_lst = [str(y) for y in range(2015, 2025)]
  # month_lst = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
  year_lst = ['2024']
  month_lst = ["12"]

  # for variable_idx in range(len(variable_lst)):
  #   for year in year_lst:
  #     for month in month_lst:
  #       variable = variable_lst[variable_idx]
  #       reanalysis_file = os.path.join(era5_root, f"single_level/reanalysis/{year}/{month}/{variable}.nc")
  #       ensemble_spread_file = os.path.join(era5_root, f"single_level/ensemble_spread/{year}/{month}/{variable}.nc")
  #       if not (os.path.exists(reanalysis_file) and os.path.exists(ensemble_spread_file)):
  #         continue
  #       interpolated_ensemble_spread_file = os.path.join(era5_root, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc")
  #       os.makedirs(os.path.dirname(interpolated_ensemble_spread_file), exist_ok=True)

  #       print(f"[INFO] Interpolating {year}-{month} {variable} ......")
  #       interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_spread_file, interpolated_ensemble_spread_file)
  
  # variable_lst = [
  #   "temperature",
  #   "u_component_of_wind",
  #   "v_component_of_wind",
  #   "specific_humidity",
  #   "geopotential",
  # ]
  # pressure_level_lst = [
  #   "50",
  #   "100",
  #   "150",
  #   "200",
  #   "250",
  #   "300",
  #   "400",
  #   "500",
  #   "600",
  #   "700",
  #   "850",
  #   "925",
  #   "1000",
  # ]
  pressure_level_lst = [
    "1", "2", "3",
    "5", "7", "10",
    "20", "30", "50",
    "70", "100", "125",
    "150", "175", "200",
    "225", "250", "300",
    "350", "400", "450",
    "500", "550", "600",
    "650", "700", "750",
    "775", "800", "825",
    "850", "875", "900",
    "925", "950", "975",
    "1000"
  ]

  for variable_idx in range(len(variable_lst)):
    for year in year_lst:
      for month in month_lst:
        for pressure_level in pressure_level_lst:
          variable = variable_lst[variable_idx]
          reanalysis_file = os.path.join(era5_root, f"pressure_level/reanalysis/{year}/{month}/{pressure_level}/{variable}.nc")
          ensemble_spread_file = os.path.join(era5_root, f"pressure_level/ensemble_spread/{year}/{month}/{pressure_level}/{variable}.nc")
          if not (os.path.exists(reanalysis_file) and os.path.exists(ensemble_spread_file)):
            continue
          interpolated_ensemble_spread_file = os.path.join(era5_root, f"pressure_level/interpolated_ensemble_spread/{year}/{month}/{pressure_level}/{variable}.nc")
          os.makedirs(os.path.dirname(interpolated_ensemble_spread_file), exist_ok=True)

          print(f"[INFO] Interpolating {year}-{month} {variable} @ {pressure_level} hPa ......")
          interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_spread_file, interpolated_ensemble_spread_file)

if __name__ == "__main__":
  main()
