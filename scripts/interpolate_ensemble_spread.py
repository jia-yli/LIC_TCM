import os
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
    ds_output = ds_output.astype(data_source.dtype)

    ds_output.to_netcdf(output_file)

    return True


if __name__ == '__main__':
  variable_lst = [
    "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "2m_temperature",
    # "total_precipitation"
  ]
  era5_root = "/capstor/scratch/cscs/ljiayong/datasets/ERA5_large"
  year_lst = [str(y) for y in range(2015, 2025)]
  month_lst = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

  for variable_idx in range(len(variable_lst)):
    for year in year_lst:
      for month in month_lst:
        variable = variable_lst[variable_idx]
        reanalysis_file = os.path.join(era5_root, f"single_level/reanalysis/{year}/{month}/{variable}.nc")
        ensemble_spread_file = os.path.join(era5_root, f"single_level/ensemble_spread/{year}/{month}/{variable}.nc")
        interpolated_ensemble_spread_file = os.path.join(era5_root, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc")
        os.makedirs(os.path.dirname(interpolated_ensemble_spread_file), exist_ok=True)

        print(f"[INFO] Interpolating {year}-{month} {variable} ......")
        interpolate_ensemble_to_reanalysis(reanalysis_file, ensemble_spread_file, interpolated_ensemble_spread_file)

  # mp
  # with mp.Pool(processes=32) as pool:  # Adjust processes as needed
  #   results = []
  #   for variable_idx in range(len(variable_lst)):
  #     for year in year_lst:
  #       for month in month_lst:
  #         variable = variable_lst[variable_idx]
  #         reanalysis_file = os.path.join(era5_root, f"single_level/reanalysis/{year}/{month}/{variable}.nc")
  #         ensemble_spread_file = os.path.join(era5_root, f"single_level/ensemble_spread/{year}/{month}/{variable}.nc")
  #         interpolated_ensemble_spread_file = os.path.join(era5_root, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.nc")
  #         os.makedirs(os.path.dirname(interpolated_ensemble_spread_file), exist_ok=True)

  #         print(f"[INFO] Starting Interpolating {year}-{month} {variable} ......")
  #         results.append(
  #           pool.apply_async(
  #             interpolate_ensemble_to_reanalysis,
  #             (reanalysis_file, ensemble_spread_file, interpolated_ensemble_spread_file),
  #           )
  #         )
  #   for res in tqdm(results):
  #     res.get()
