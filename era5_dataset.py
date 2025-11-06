import random
import os
import itertools
import tempfile
import xarray as xr
from joblib import Memory
from multiprocessing import Pool

import numpy as np
import pandas as pd

memory = Memory(location="/iopsstor/scratch/cscs/ljiayong/cache/era5_dataset", verbose=0)
# memory.clear()

def reindex_dataset(ds, interval):
  new_time = pd.date_range(ds.valid_time[0].values, ds.valid_time[-1].values, freq=interval)
  return ds.sel(valid_time=new_time)

@memory.cache
def load_single_level_file(era5_root, product_type, year, month, variable, interval=None):
  data_file = os.path.join(era5_root, f"single_level/{product_type}/{year}/{month}/{variable}.nc")
  dataset = xr.open_dataset(data_file, engine="netcdf4")
  assert len(dataset.data_vars) == 1
  var_short = list(dataset.data_vars)[0]

  # print(f"Load {var_short}: {dataset.data_vars}")
  if interval is not None:
    dataset = reindex_dataset(dataset, interval)
  data = dataset[var_short].values
  metadata = {
    "valid_time": dataset.valid_time.values,
    "latitude": dataset.latitude.values,
    "longitude": dataset.longitude.values,
  }
  return data, metadata
  
def load_single_level_data(era5_root, year, month, variable, interval=None):
  reanalysis_data, reanalysis_metadata = load_single_level_file(era5_root, "reanalysis", year, month, variable, interval)
  interpolated_ensemble_spread_data, interpolated_ensemble_spread_metadata = load_single_level_file(era5_root, "interpolated_ensemble_spread", year, month, variable, interval)

  return reanalysis_data, interpolated_ensemble_spread_data

def warmup_worker(era5_root, year, month, variable):
  """Worker function for parallel warmup loading with validation"""
  reanalysis_data, interpolated_ensemble_spread_data = load_single_level_data(era5_root, year, month, variable)
  assert reanalysis_data.shape == interpolated_ensemble_spread_data.shape, \
    f"Shape mismatch for {variable} {year}/{month}: {reanalysis_data.shape} vs {interpolated_ensemble_spread_data.shape}"
  num_samples = reanalysis_data.shape[0]
  return num_samples

def calc_padding(shape, factor):
  h, w = shape[-2], shape[-1]
  new_h = (h + factor - 1) // factor * factor
  new_w = (w + factor - 1) // factor * factor
  padding_left = (new_w - w) // 2
  padding_right = new_w - w - padding_left
  padding_top = (new_h - h) // 2
  padding_bottom = new_h - h - padding_top
  return padding_left, padding_right, padding_top, padding_bottom

def pad_np(x, padding, v=np.nan):
  padding_left, padding_right, padding_top, padding_bottom = padding
  pad_width = [(0, 0)] * x.ndim
  pad_width[-2] = (padding_top, padding_bottom)
  pad_width[-1] = (padding_left, padding_right)
  x_padded = np.pad(x, pad_width, mode="constant", constant_values=v)
  return x_padded

def crop_np(x, padding):
  padding_left, padding_right, padding_top, padding_bottom = padding
  return x[..., padding_top:x.shape[-2]-padding_bottom, padding_left:x.shape[-1]-padding_right]

class Era5ReanalysisDatasetSingleLevel:
  def __init__(self, 
    variable, 
    batch_size, 
    patch_size=256, # size of sampled patches; -1 means full image for train split only
    padding_factor=128, # pad to multiple of this factor, before random cropping, no pad if = 1
    split='train', 
    n_files_per_load=8):
    # configs
    self.variable = variable
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.padding_factor = padding_factor
    self.padding = None
    self.split = split
    self.n_files_per_load = n_files_per_load
    self.era5_root = "/capstor/scratch/cscs/ljiayong/datasets/ERA5_large"
    if split == 'train':
      # self.year_lst = [str(y) for y in range(2015, 2023)]
      self.year_lst = [str(y) for y in range(2021, 2023)]
    elif split == 'valid':
      self.year_lst = [str(y) for y in range(2023, 2024)]
    elif split == 'test':
      self.year_lst = [str(y) for y in range(2024, 2025)]
    else:
      raise ValueError(f"Unsupported dataset split: {split}")
    self.month_lst = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    self.existing_files = self.check_data_availability()
    self.total_samples = self.warmup_cache(num_processes=64)

    # states
    self._reset_file_iterator()
    self._reset_data_buffer()
  
  def check_data_availability(self):
    print(f"[INFO] Checking data availability for variable: {self.variable}, missing files will be listed below.")
    existing = []
    missing = []
    for year in self.year_lst:
      for month in self.month_lst:
        reanalysis_file = os.path.join(self.era5_root, f"single_level/reanalysis/{year}/{month}/{self.variable}.nc")
        interpolated_ensemble_spread_file = os.path.join(self.era5_root, f"single_level/interpolated_ensemble_spread/{year}/{month}/{self.variable}.nc")
        reanalysis_file_exists = os.path.exists(reanalysis_file)
        interpolated_ensemble_spread_file_exists = os.path.exists(interpolated_ensemble_spread_file)

        if (not reanalysis_file_exists) or (not interpolated_ensemble_spread_file_exists):
          missing.append({
            'year': year,
            'month': month,
            'reanalysis_file_exists': reanalysis_file_exists,
            'interpolated_ensemble_spread_file_exists': interpolated_ensemble_spread_file_exists
          })
        else:
          existing.append((year, month))

    print(pd.DataFrame(missing))
    return existing
  
  def warmup_cache(self, num_processes):
    print(f"[INFO] Warming up cache for variable: {self.variable}")
    with Pool(processes=num_processes) as pool:
      results = pool.starmap(warmup_worker, [(self.era5_root, year, month, self.variable) for year, month in self.existing_files])
    total_samples = sum(results)
    print(f"[INFO] Finished warming up cache for variable: {self.variable}")
    print(f"[INFO] Total samples: {total_samples}")
    return total_samples
  
  def _reset_file_iterator(self):
    """Reset the file iterator (shuffle for train, sequential for valid/test)"""
    self.file_list = self.existing_files.copy()
    if self.split == 'train':
      random.shuffle(self.file_list)
    self.file_index = 0
  
  def _reset_data_buffer(self):
    """Reset the data buffer that holds concatenated data from N files"""
    self.current_data = None
    self.current_spread = None
    self.n_samples = 0
    self.current_start_idx = 0
  
  def _load_n_files(self):
    """Load next N files, concatenate, and shuffle (for train)"""
    n_files_to_load = min(self.n_files_per_load, len(self.file_list) - self.file_index)
    
    if n_files_to_load == 0:
      return None, None, True  # all files seen

    data_list = []
    spread_list = []
    for i in range(n_files_to_load):
      year, month = self.file_list[self.file_index]
      self.file_index += 1
      data, spread = load_single_level_data(self.era5_root, year, month, self.variable)
      data_list.append(data)
      spread_list.append(spread)

    # Concatenate all loaded files
    concat_data = np.concatenate(data_list, axis=0)
    concat_spread = np.concatenate(spread_list, axis=0)

    if self.padding is None:
      self.padding = calc_padding(concat_data.shape, self.padding_factor)
      print(f"[INFO] Calculated padding: {self.padding} for data shape: {concat_data.shape} with factor: {self.padding_factor}")
    concat_data = pad_np(concat_data, self.padding, v=np.nan)
    concat_spread = pad_np(concat_spread, self.padding, v=np.nan)

    n_samples = concat_data.shape[0]

    # Shuffle the concatenated data (for train) while maintaining correspondence
    if self.split == 'train':
      shuffle_indices = np.random.permutation(n_samples)
      concat_data = concat_data[shuffle_indices]
      concat_spread = concat_spread[shuffle_indices]

    assert self.file_index <= len(self.file_list)
    all_files_seen = self.file_index >= len(self.file_list)
    return concat_data, concat_spread, all_files_seen

  def _ensure_data_available(self):
    """Ensure there is data in the buffer, load more if needed"""
    if self.current_data is None or self.current_start_idx >= self.n_samples:
      # Need to load more data
      new_data, new_spread, all_files_seen = self._load_n_files()
      
      if new_data is None:
        return True
      
      if self.current_data is not None and self.current_start_idx < self.n_samples:
        # Concat remaining data with new data
        self.current_data = np.concatenate([self.current_data[self.current_start_idx:], new_data], axis=0)
        self.current_spread = np.concatenate([self.current_spread[self.current_start_idx:], new_spread], axis=0)
      else:
        # Use new data
        self.current_data = new_data
        self.current_spread = new_spread
      
      self.n_samples = self.current_data.shape[0]
      self.current_start_idx = 0
      
      return all_files_seen
    
    return False

  def sample_batch(self):
    batch_data = []
    batch_spread = []
    current_batch_size = 0
    is_epoch_finished = False
    
    while current_batch_size < self.batch_size:
      # Check if we need more data
      n_remaining = self.n_samples - self.current_start_idx if self.current_data is not None else 0
      
      # Load next N files if current remaining is not enough for a full batch and more files available
      if n_remaining < (self.batch_size - current_batch_size) and self.file_index < len(self.file_list):
        self._ensure_data_available()
      
      # If no data available at all, break
      if self.current_data is None or self.n_samples == 0:
        break
      
      # Select batch
      n_remaining = self.n_samples - self.current_start_idx
      n_take = min(n_remaining, self.batch_size - current_batch_size)
      selected_idx = np.arange(self.current_start_idx, self.current_start_idx + n_take)
      self.current_start_idx += n_take
      
      selected_data = self.current_data[selected_idx]
      selected_spread = self.current_spread[selected_idx]
      
      # Apply random crop per sample if patch_size > 0
      if (self.patch_size > 0) and self.split == 'train':
        h, w = selected_data.shape[-2:]
        assert h >= self.patch_size and w >= self.patch_size

        h_start_idx = np.random.randint(0, h - self.patch_size + 1, size = n_take)
        w_start_idx = np.random.randint(0, w - self.patch_size + 1, size = n_take)

        # idx 2D shape [n_take, patch_size]
        h_idx = h_start_idx[:, np.newaxis] + np.arange(self.patch_size)[np.newaxis, :]
        w_idx = w_start_idx[:, np.newaxis] + np.arange(self.patch_size)[np.newaxis, :]

        selected_data = selected_data[np.arange(n_take)[:, np.newaxis, np.newaxis], h_idx[:, :, np.newaxis], w_idx[:, np.newaxis, :]]
        selected_spread = selected_spread[np.arange(n_take)[:, np.newaxis, np.newaxis], h_idx[:, :, np.newaxis], w_idx[:, np.newaxis, :]]

      batch_data.append(selected_data)
      batch_spread.append(selected_spread)
      current_batch_size += n_take
      
      # If no more files and buffer exhausted, we're done
      if self.file_index >= len(self.file_list) and self.current_start_idx >= self.n_samples:
        is_epoch_finished = True
        # Reset for next epoch
        self._reset_file_iterator()
        self._reset_data_buffer()
        break

    batch_data_array = np.concatenate(batch_data, axis=0)  # shape [b, h, w]
    batch_spread_array = np.concatenate(batch_spread, axis=0)  # shape [b, h, w]
    
    return batch_data_array, batch_spread_array, is_epoch_finished

# ------------- Helper functions for train/valid loop + tests -----------------

def run_one_epoch(dataset: Era5ReanalysisDatasetSingleLevel, split_name: str):
  """
  Run through one full epoch using sample_batch() and perform checks:
    1) Every batch except possibly the last has full batch_size.
    2) Total number of samples == dataset.total_samples.
  """
  print(f"\n[RUN] {split_name} epoch with batch_size={dataset.batch_size}, patch_size={dataset.patch_size}")
  total_seen = 0
  batch_idx = 0
  full_batches = 0
  partial_batches = 0

  while True:
    batch_data, batch_spread, is_epoch_finished = dataset.sample_batch()
    bsz = batch_data.shape[0]
    batch_idx += 1
    total_seen += bsz

    # 1. Check full batch except last
    if bsz == dataset.batch_size:
      full_batches += 1
    else:
      partial_batches += 1
      # if not finished, this is a bug
      assert is_epoch_finished, (
        f"[ERROR] Non-final partial batch in {split_name} split: "
        f"batch {batch_idx} has size {bsz} but epoch not finished."
      )

    print(f"[{split_name.upper()}] Batch {batch_idx:03d}: shape={batch_data.shape}, is_epoch_finished={is_epoch_finished}")

    if is_epoch_finished:
      break

  print(f"[CHECK] {split_name}: full_batches={full_batches}, partial_batches={partial_batches}")
  # 2. Check total number of samples == total_samples
  if dataset.total_samples > 0:
    assert total_seen == dataset.total_samples, (
      f"[ERROR] {split_name}: total_seen={total_seen} != dataset.total_samples={dataset.total_samples}"
    )
    print(f"[CHECK OK] {split_name}: total_seen == dataset.total_samples == {total_seen}")
  else:
    print(f"[WARN] {split_name}: dataset.total_samples == 0 (probably no files found).")

  return total_seen


def test_random_patching_logic():
  """
  3. Test patching logic:
     - patch equals the same region of the original (no-patch) image;
     - patch locations are random (at least some call differences).
  This uses a synthetic array but the logic is identical to sample_batch().
  """
  print("\n[TEST] Random patching logic")

  # synthetic data: shape [B, H, W]
  B, H, W = 4, 10, 12
  patch_size = 5
  base = np.arange(B * H * W).reshape(B, H, W)

  # Fix seed for reproducibility of this test
  np.random.seed(0)

  # First patch sample using the same logic as in sample_batch()
  h_start_1 = np.random.randint(0, H - patch_size + 1, size=B)
  w_start_1 = np.random.randint(0, W - patch_size + 1, size=B)
  h_idx_1 = h_start_1[:, None] + np.arange(patch_size)[None, :]
  w_idx_1 = w_start_1[:, None] + np.arange(patch_size)[None, :]

  patch_1 = base[np.arange(B)[:, None, None], h_idx_1[:, :, None], w_idx_1[:, None, :]]

  # Check that each patch exactly equals region from original
  for i in range(B):
    region = base[i, h_start_1[i]:h_start_1[i]+patch_size, w_start_1[i]:w_start_1[i]+patch_size]
    assert np.array_equal(patch_1[i], region), "[ERROR] Patch does not match original region."
  print("[CHECK OK] Patch equals region of original (no-patch) image.")

  # Second sample to check randomness
  np.random.seed(1)  # change seed
  h_start_2 = np.random.randint(0, H - patch_size + 1, size=B)
  w_start_2 = np.random.randint(0, W - patch_size + 1, size=B)

  same_positions = np.all(h_start_1 == h_start_2) and np.all(w_start_1 == w_start_2)
  assert not same_positions, "[ERROR] Patch positions are deterministic; expected random."
  print("[CHECK OK] Patch locations differ for different RNG seeds -> patching is random.")


# -------------------------- Example usage / tests ----------------------------

if __name__ == "__main__":
  # Make everything deterministic for this demo run
  random.seed(0)
  np.random.seed(0)

  # Example: create train / valid / test datasets
  train_dataset = Era5ReanalysisDatasetSingleLevel(
    variable="2m_temperature",
    batch_size=64,
    patch_size=256,
    padding_factor=128,
    split="train",
    n_files_per_load=4,
  )

  valid_dataset = Era5ReanalysisDatasetSingleLevel(
    variable="2m_temperature",
    batch_size=32,
    patch_size=-1,
    padding_factor=1,
    split="valid",
    n_files_per_load=4,
  )

  # Run a "train-valid" style loop, but only over the data (no model training)
  print("\n========== RUN TRAIN EPOCH (data-only) ==========")
  train_seen = run_one_epoch(train_dataset, "train")

  print("\n========== RUN VALID EPOCH (data-only) ==========")
  valid_seen = run_one_epoch(valid_dataset, "valid")

  print(f"\n[SUMMARY] Samples per epoch -> train: {train_seen}, valid: {valid_seen}")

  # 3. Test the patching logic separately on synthetic data
  test_random_patching_logic()

  print("\n[ALL TESTS COMPLETED]")