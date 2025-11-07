import os
import bisect
import random
import xarray as xr
from multiprocessing import Pool

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def load_single_level_file(era5_root, product_type, year, month, variable):
  data_file = os.path.join(era5_root, f"single_level/{product_type}/{year}/{month}/{variable}.nc")
  dataset = xr.open_dataset(data_file, engine="netcdf4")
  assert len(dataset.data_vars) == 1
  var_short = list(dataset.data_vars)[0]

  data = dataset[var_short].values
  metadata = {
    "valid_time": dataset.valid_time.values,
    "latitude": dataset.latitude.values,
    "longitude": dataset.longitude.values,
  }
  return data, metadata

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

class Era5ReanalysisDatasetSingleLevel(Dataset):
  def __init__(self, 
    variable, 
    split='train',
    patch_size=256,      # size of sampled patches; -1 means full image
    padding_factor=128,  # pad to multiple of this factor; 1 => no pad
  ):
    super().__init__()

    # configs
    self.variable = variable
    self.patch_size = patch_size
    self.padding_factor = padding_factor
    self.padding = None
    self.split = split

    self.era5_root = "/capstor/scratch/cscs/ljiayong/datasets/ERA5_large"
    self.npy_root = "/iopsstor/scratch/cscs/ljiayong/cache/era5_npy"

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

    self.build_npy_storage(num_processes=64)

  def check_data_availability(self):
    print(f"[INFO] Checking data availability for variable: {self.variable}, split: {self.split}")
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

    if len(missing) > 0:
      print(f"[INFO] Missing .nc files detected for variable: {self.variable}")
      print(pd.DataFrame(missing))
    else:
      print("[INFO] No missing .nc files detected.")
    return existing

  @staticmethod
  def _build_npy_storage(era5_root, npy_root, year, month, variable):
    os.makedirs(npy_root, exist_ok=True)
    reanalysis_path = os.path.join(npy_root, f"single_level/reanalysis/{year}/{month}/{variable}.npy")
    spread_path = os.path.join(npy_root, f"single_level/interpolated_ensemble_spread/{year}/{month}/{variable}.npy")

    # make sure subdirs exist
    os.makedirs(os.path.dirname(reanalysis_path), exist_ok=True)
    os.makedirs(os.path.dirname(spread_path), exist_ok=True)

    # If already converted, reuse existing .npy and just read shape
    if os.path.exists(reanalysis_path) and os.path.exists(spread_path):
      arr = np.load(reanalysis_path, mmap_mode="r")  # [Ntime, H, W]
      n_time, H, W = arr.shape
      return year, month, reanalysis_path, spread_path, n_time, H, W

    # otherwise convert from .nc
    reanalysis_data, _ = load_single_level_file(era5_root, "reanalysis", year, month, variable)
    spread_data, _ = load_single_level_file(era5_root, "interpolated_ensemble_spread", year, month, variable)

    assert reanalysis_data.shape == spread_data.shape, \
      f"Shape mismatch for {variable} {year}/{month}: {reanalysis_data.shape} vs {spread_data.shape}"

    n_time, H, W = reanalysis_data.shape

    np.save(reanalysis_path, reanalysis_data)
    np.save(spread_path, spread_data)

    return year, month, reanalysis_path, spread_path, n_time, H, W

  def build_npy_storage(self, num_processes):
    print(f"[INFO] Building .npy storage for variable: {self.variable}, split: {self.split}, ...")
    with Pool(processes=num_processes) as pool:
      results = pool.starmap(
        Era5ReanalysisDatasetSingleLevel._build_npy_storage, 
        [(self.era5_root, self.npy_root, year, month, self.variable) for year, month in self.existing_files]
      )

    # reset index
    self.reanalysis_files = []
    self.spread_files = []
    self.file_lengths = []
    self.cumulative_starts = []
    self.total_samples = 0
    self.padding = None

    # parse worker results to build index
    for (year, month, reanalysis_path, spread_path, n_time, H, W) in results:
      assert n_time > 0, f"Invalid time dimension for {year}/{month}: {n_time}"

      self.reanalysis_files.append(reanalysis_path)
      self.spread_files.append(spread_path)
      self.file_lengths.append(n_time)
      self.cumulative_starts.append(self.total_samples)
      self.total_samples += n_time

      if self.padding is None:
        self.padding = calc_padding((n_time, H, W), self.padding_factor)
        print(
          f"[INFO] Calculated padding for split={self.split}: {self.padding} "
            f"with factor: {self.padding_factor}, original spatial shape: {(H, W)}"
        )

    n_files = len(self.reanalysis_files)
    print(f"[INFO] Built .npy index for split={self.split}, variable={self.variable}: "
          f"{n_files} files, total_samples={self.total_samples}")

    # memmap slots
    self._reanalysis_arrays = [None] * n_files
    self._spread_arrays = [None] * n_files

    print(f"[INFO] Built .npy storage for variable: {self.variable}, split: {self.split}")
    return

  def __len__(self):
    return self.total_samples

  def _get_arrays(self, file_idx: int):
    """
    Lazily open memmap arrays for a given file index (per DataLoader worker).
    """
    arr_r = self._reanalysis_arrays[file_idx]
    arr_s = self._spread_arrays[file_idx]

    if arr_r is None:
      arr_r = np.load(self.reanalysis_files[file_idx], mmap_mode="r")
      self._reanalysis_arrays[file_idx] = arr_r

    if arr_s is None:
      arr_s = np.load(self.spread_files[file_idx], mmap_mode="r")
      self._spread_arrays[file_idx] = arr_s

    return arr_r, arr_s

  def _locate_idx(self, idx: int):
    """
    Map a global index idx -> (file_idx, local_idx).
    """
    if idx < 0:
      idx += self.total_samples
    if idx < 0 or idx >= self.total_samples:
      raise IndexError(idx)

    file_idx = bisect.bisect_right(self.cumulative_starts, idx) - 1
    file_start = self.cumulative_starts[file_idx]
    local_idx = idx - file_start
    return file_idx, local_idx

  def _random_patch_single(self, x):
    """
    Random crop a single [H, W] array if patch_size > 0.
    """
    h, w = x.shape[-2], x.shape[-1]
    assert h >= self.patch_size and w >= self.patch_size, \
      f"Patch size {self.patch_size} is larger than field {(h, w)}."

    h_start = np.random.randint(0, h - self.patch_size + 1)
    w_start = np.random.randint(0, w - self.patch_size + 1)
    return x[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]

  def __getitem__(self, idx):
    file_idx, local_idx = self._locate_idx(idx)
    arr_r, arr_s = self._get_arrays(file_idx)

    # single sample [H, W]
    data = arr_r[local_idx]
    spread = arr_s[local_idx]

    # padding
    data = pad_np(data, self.padding, v=np.nan)
    spread = pad_np(spread, self.padding, v=np.nan)

    # random patch for train if enabled; valid/test: full data
    if self.split == 'train' and self.patch_size > 0:
      data = self._random_patch_single(data)
      spread = self._random_patch_single(spread)

    # convert to torch tensors
    data = torch.from_numpy(np.asarray(data, dtype=np.float32))
    spread = torch.from_numpy(np.asarray(spread, dtype=np.float32))

    return data, spread


def run_one_epoch(dataloader, split_name: str, total_expected: int):
  """
  Run through one full epoch using a PyTorch DataLoader and perform checks:
    1) Every batch except possibly the last has full batch_size.
    2) Total number of samples == total_expected (dataset.__len__()).
  """
  print(f"\n[RUN] {split_name} epoch with batch_size={dataloader.batch_size}")
  total_seen = 0
  batch_idx = 0
  full_batches = 0
  partial_batches = 0
  bs = dataloader.batch_size

  for batch_idx, (batch_data, batch_spread) in enumerate(dataloader, start=1):
    bsz = batch_data.shape[0]
    total_seen += bsz

    if bs is not None and bsz == bs:
      full_batches += 1
    else:
      partial_batches += 1

    print(f"[{split_name.upper()}] Batch {batch_idx:03d}: "
          f"shape={tuple(batch_data.shape)}, "
          f"is_partial={(bs is None) or (bsz != bs)}")

  print(f"[CHECK] {split_name}: full_batches={full_batches}, partial_batches={partial_batches}")
  if total_expected > 0:
    assert total_seen == total_expected, (
      f"[ERROR] {split_name}: total_seen={total_seen} != total_expected={total_expected}"
    )
    print(f"[CHECK OK] {split_name}: total_seen == total_expected == {total_seen}")
  else:
    print(f"[WARN] {split_name}: total_expected == 0 (probably no .npy files found).")

  return total_seen

if __name__ == "__main__":
  import time
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)

  variable = "2m_temperature"

  # Example: create train / valid / test datasets
  train_dataset = Era5ReanalysisDatasetSingleLevel(
    variable=variable,
    split="train",
    patch_size=256,
    padding_factor=128,
  )

  valid_dataset = Era5ReanalysisDatasetSingleLevel(
    variable=variable,
    split="valid",
    patch_size=-1,
    padding_factor=1,
  )


  train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=16,
    drop_last=False,
  )

  valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=16,
    drop_last=False,
  )

  print("\n========== RUN TRAIN EPOCH (data-only) ==========")
  start_time = time.time()
  train_seen = run_one_epoch(train_loader, "train", len(train_dataset))
  train_seen = run_one_epoch(train_loader, "train", len(train_dataset))
  end_time = time.time()
  print(f"[INFO] Train epoch time: {end_time - start_time:.2f} seconds")

  print("\n========== RUN VALID EPOCH (data-only) ==========")
  start_time = time.time()
  valid_seen = run_one_epoch(valid_loader, "valid", len(valid_dataset))
  valid_seen = run_one_epoch(valid_loader, "valid", len(valid_dataset))
  end_time = time.time()
  print(f"[INFO] Valid epoch time: {end_time - start_time:.2f} seconds")

  print(f"\n[SUMMARY] Samples per epoch -> train: {train_seen}, valid: {valid_seen}")

  print("\n[ALL TESTS COMPLETED]")
