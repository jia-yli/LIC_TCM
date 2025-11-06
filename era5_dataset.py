import random
import os
import itertools
import tempfile
import xarray as xr
from joblib import Memory
import threading
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
    n_files_per_load=8,
    loader_mode='thread', # 'sync', 'thread', or 'process'
  ):
    # configs
    self.variable = variable
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.padding_factor = padding_factor
    self.padding = None
    self.split = split
    self.n_files_per_load = n_files_per_load
    self.loader_mode = loader_mode

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

  @staticmethod
  def warmup_worker(era5_root, year, month, variable):
    """Worker function for parallel warmup loading with validation"""
    reanalysis_data, interpolated_ensemble_spread_data = load_single_level_data(era5_root, year, month, variable)
    assert reanalysis_data.shape == interpolated_ensemble_spread_data.shape, \
      f"Shape mismatch for {variable} {year}/{month}: {reanalysis_data.shape} vs {interpolated_ensemble_spread_data.shape}"
    num_samples = reanalysis_data.shape[0]
    return num_samples

  def warmup_cache(self, num_processes):
    print(f"[INFO] Warming up cache for variable: {self.variable}")
    with Pool(processes=num_processes) as pool:
      results = pool.starmap(self.warmup_worker, [(self.era5_root, year, month, self.variable) for year, month in self.existing_files])
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
    """Reset the data buffer and clean up any in-flight prefetch workers."""
    # --- clean up previous prefetch state if it exists ---
    # use getattr because __init__ calls this before attributes exist
    prev_thread = getattr(self, "_prefetch_thread", None)
    if prev_thread is not None:
      prev_thread.join()
    
    prev_pool = getattr(self, "_prefetch_pool", None)
    if prev_pool is not None:
      try:
        prev_pool.terminate()
      except Exception:
        pass
      prev_pool.join()

    # --- main buffer state ---
    self.current_data = None
    self.current_spread = None
    self.n_samples = 0
    self.current_start_idx = 0

    # --- prefetch loader state ---
    self._prefetch_loading = False
    self._prefetch_thread = None
    self._prefetch_pool = None
    self._prefetch_async_result = None
    self._prefetch_data = None      # used for thread mode
    self._prefetch_spread = None    # used for thread mode
    self._prefetch_all_files_seen = False

    if self.loader_mode in ("thread", "process"):
      file_list = getattr(self, "file_list", None)
      if file_list is not None and len(file_list) > 0:
        # file_index is set by _reset_file_iterator(), which is always
        # called before _reset_data_buffer()
        self._start_async_load()
  
  @staticmethod
  def load_files_worker(
    era5_root, 
    files_to_load, 
    variable, 
    padding, 
    padding_factor, 
    split
  ):
    data_list = []
    spread_list = []

    for year, month in files_to_load:
      data, spread = load_single_level_data(era5_root, year, month, variable)
      data_list.append(data)
      spread_list.append(spread)

    concat_data = np.concatenate(data_list, axis=0)
    concat_spread = np.concatenate(spread_list, axis=0)

    # padding
    if padding is None:
      padding = calc_padding(concat_data.shape, padding_factor)

    concat_data = pad_np(concat_data, padding, v=np.nan)
    concat_spread = pad_np(concat_spread, padding, v=np.nan)

    n_samples = concat_data.shape[0]

    # shuffle for train
    if split == 'train':
      shuffle_indices = np.random.permutation(n_samples)
      concat_data = concat_data[shuffle_indices]
      concat_spread = concat_spread[shuffle_indices]

    return concat_data, concat_spread, padding

  def _load_n_files(self):
    """Load next N files, concatenate, and shuffle (for train)"""
    n_files_to_load = min(self.n_files_per_load, len(self.file_list) - self.file_index)
    
    if n_files_to_load == 0:
      return None, None, True  # all files seen

    files_to_load = self.file_list[self.file_index:self.file_index + n_files_to_load]
    self.file_index += n_files_to_load

    concat_data, concat_spread, padding = self.load_files_worker(
      self.era5_root,
      files_to_load,
      self.variable,
      self.padding,
      self.padding_factor,
      self.split
    )

    if self.padding is None:
      self.padding = padding
      print(f"[INFO] Calculated padding: {self.padding} with factor: {self.padding_factor}, padded data shape: {concat_data.shape}")

    assert self.file_index <= len(self.file_list)
    all_files_seen = self.file_index >= len(self.file_list)
    return concat_data, concat_spread, all_files_seen
  
  def _start_async_load(self):
    """Kick off async load of the *next* chunk (thread or process)."""
    # only async modes use prefetch
    if self.loader_mode not in ("thread", "process"):
      return

    # if something is already loading or we have a prefetched result waiting, do nothing
    if self._prefetch_loading:
      return
    if self.loader_mode == "thread" and self._prefetch_data is not None:
      return
    if self.loader_mode == "process" and self._prefetch_async_result is not None:
      return

    # no more files to schedule
    if self.file_index >= len(self.file_list):
      return

    n_files_to_load = min(self.n_files_per_load, len(self.file_list) - self.file_index)
    if n_files_to_load == 0:
      return

    files_to_load = self.file_list[self.file_index:self.file_index + n_files_to_load]
    self.file_index += n_files_to_load
    self._prefetch_all_files_seen = (self.file_index >= len(self.file_list))
    self._prefetch_loading = True

    # THREAD MODE: load into _prefetch_data / _prefetch_spread
    if self.loader_mode == "thread":
      def worker():
        try:
          concat_data, concat_spread, padding_used = self.load_files_worker(
            self.era5_root,
            files_to_load,
            self.variable,
            self.padding,
            self.padding_factor,
            self.split
          )
          if self.padding is None:
            self.padding = padding_used
            print(f"[INFO] Calculated padding: {self.padding} for data shape: {concat_data.shape} with factor: {self.padding_factor}")
          self._prefetch_data = concat_data
          self._prefetch_spread = concat_spread
        finally:
          self._prefetch_loading = False

      self._prefetch_thread = threading.Thread(target=worker, daemon=True)
      self._prefetch_thread.start()

    # PROCESS MODE: load into an async result (data is retrieved in _ensure_data_available)
    elif self.loader_mode == "process":
      if self._prefetch_pool is not None:
        # just in case, clean up previous pool
        try:
          self._prefetch_pool.terminate()
        except Exception:
          pass
        self._prefetch_pool.join()

      self._prefetch_pool = Pool(processes=1)
      self._prefetch_async_result = self._prefetch_pool.apply_async(
        Era5ReanalysisDatasetSingleLevel.load_files_worker,
        args=(
          self.era5_root,
          files_to_load,
          self.variable,
          self.padding,
          self.padding_factor,
          self.split
        )
      )

  def _ensure_data_available(self):
    """
    Ensure there is at least one sample available in `current_data`.
    If the current chunk is exhausted, load the next chunk (sync or async).
    Returns True *only* when there is no more data at all (epoch finished).
    """
    # If we still have data in the current buffer, nothing to do
    if (self.current_data is not None) and (self.current_start_idx < self.n_samples):
      return False  # not finished

    # Drop exhausted buffer
    self.current_data = None
    self.current_spread = None
    self.n_samples = 0
    self.current_start_idx = 0

    new_data = None
    new_spread = None

    # ---- SYNC MODE ----
    if self.loader_mode == "sync":
      if self.file_index < len(self.file_list):
        new_data, new_spread, _ = self._load_n_files()
      # else: no files left -> new_data stays None

    # ---- ASYNC MODES ----
    else:
      # 1) Try to consume any prefetched chunk first
      if self.loader_mode == "thread":
        if self._prefetch_thread is not None:
          self._prefetch_thread.join()
          self._prefetch_thread = None
        self._prefetch_loading = False

        if self._prefetch_data is not None:
          new_data = self._prefetch_data
          new_spread = self._prefetch_spread
          # after consuming, clear the prefetched buffers
          self._prefetch_data = None
          self._prefetch_spread = None

      elif self.loader_mode == "process":
        if self._prefetch_async_result is not None:
          concat_data, concat_spread, padding_used = self._prefetch_async_result.get()
          # clean up pool
          self._prefetch_pool.close()
          self._prefetch_pool.join()
          self._prefetch_pool = None
          self._prefetch_async_result = None
          self._prefetch_loading = False

          if self.padding is None:
            self.padding = padding_used
            print(f"[INFO] Calculated padding: {self.padding} for data shape: {concat_data.shape} with factor: {self.padding_factor}")

          new_data = concat_data
          new_spread = concat_spread

      # 2) If no prefetched chunk was available, fall back to blocking load
      if (new_data is None) and (self.file_index < len(self.file_list)):
        new_data, new_spread, _ = self._load_n_files()

    # If we have no new_data here, there is truly no more data
    if new_data is None:
      return True  # epoch finished

    # Set the new current chunk
    self.current_data = new_data
    self.current_spread = new_spread
    self.n_samples = self.current_data.shape[0]
    self.current_start_idx = 0

    # Start prefetch of the next chunk while we consume this one
    if self.loader_mode in ("thread", "process"):
      self._start_async_load()

    return False  # not finished yet

  
  def sample_batch(self):
    batch_data = []
    batch_spread = []
    current_batch_size = 0
    is_epoch_finished = False
    
    while current_batch_size < self.batch_size:
      # Make sure we have data in the current buffer
      no_more_data = self._ensure_data_available()
      if no_more_data:
        # No data at all (no current chunk and no further chunks)
        is_epoch_finished = True
        self._reset_file_iterator()
        self._reset_data_buffer()
        break

      # At this point we must have some data
      if self.current_data is None or self.n_samples == 0:
        is_epoch_finished = True
        self._reset_file_iterator()
        self._reset_data_buffer()
        break

      # Take as many samples as we can from the current chunk
      n_remaining = self.n_samples - self.current_start_idx
      n_take = min(n_remaining, self.batch_size - current_batch_size)

      if n_take <= 0:
        # Should not normally happen, but guard against infinite loops
        is_epoch_finished = True
        self._reset_file_iterator()
        self._reset_data_buffer()
        break

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

      # Loop continues; when this chunk is exhausted, next iteration will call
      # _ensure_data_available() again, which will swap in the next chunk (sync
      # or from prefetch) or signal epoch end.

    # If no samples were collected, signal epoch finished
    if len(batch_data) == 0:
      return None, None, True

    batch_data_array = np.concatenate(batch_data, axis=0)   # shape [b, ...]
    batch_spread_array = np.concatenate(batch_spread, axis=0)

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

    print(f"[{split_name.upper()}] Batch {batch_idx:03d}: shape={batch_data.shape}, is_epoch_finished={is_epoch_finished}, current_start_idx = {dataset.current_start_idx}, n_samples = {dataset.n_samples}")

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
  import time
  # Make everything deterministic for this demo run
  random.seed(0)
  np.random.seed(0)


  loader_mode = 'thread'  # 'sync', 'thread', or 'process'
  n_files_per_load = 4 
  # Example: create train / valid / test datasets
  train_dataset = Era5ReanalysisDatasetSingleLevel(
    variable="2m_temperature",
    batch_size=64,
    patch_size=256,
    padding_factor=128,
    split="train",
    n_files_per_load=n_files_per_load, # 1: 172.12s, 4: 176.73s, 16: 241.11s
    loader_mode=loader_mode,
  )

  valid_dataset = Era5ReanalysisDatasetSingleLevel(
    variable="2m_temperature",
    batch_size=32,
    patch_size=-1,
    padding_factor=1,
    split="valid",
    n_files_per_load=n_files_per_load, # 1: 68.67s, 4: 75.50s, 16: 113.33s
    loader_mode=loader_mode,
  )

  # Run a "train-valid" style loop, but only over the data (no model training)
  print("\n========== RUN TRAIN EPOCH (data-only) ==========")
  start_time = time.time()
  train_seen = run_one_epoch(train_dataset, "train")
  train_seen = run_one_epoch(train_dataset, "train")
  end_time = time.time()
  print(f"[INFO] Train epoch time: {end_time - start_time:.2f} seconds")
  # sync: 216.70s, thread: 176.73s, process: 304.59s

  print("\n========== RUN VALID EPOCH (data-only) ==========")
  start_time = time.time()
  valid_seen = run_one_epoch(valid_dataset, "valid")
  valid_seen = run_one_epoch(valid_dataset, "valid")
  end_time = time.time()
  print(f"[INFO] Valid epoch time: {end_time - start_time:.2f} seconds")
  # sync: 91.45s, thread:  75.50s, process: 116.31s

  print(f"\n[SUMMARY] Samples per epoch -> train: {train_seen}, valid: {valid_seen}")

  # 3. Test the patching logic separately on synthetic data
  test_random_patching_logic()

  print("\n[ALL TESTS COMPLETED]")