import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
from mmengine import Config
import pandas as pd
import multiprocessing as mp
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import itertools
warnings.filterwarnings("ignore")

def compute_psnr(a, b):
  mse = torch.mean((a - b)**2).item()
  return -10 * math.log10(mse)

def compute_msssim(a, b):
  return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
  size = out_net['x_hat'].size()
  num_pixels = size[0] * size[2] * size[3]
  return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net['likelihoods'].values()).item()

class Era5ReanalysisDataset:
  def __init__(self, variable, batch_size, patch_size=256, split='train'):
    # configs
    self.variable = variable
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.split = split
    self.era5_root = "/capstor/scratch/cscs/ljiayong/datasets/ERA5_large"
    if split == 'train':
      self.year_lst = [str(y) for y in range(2015, 2023)]
      # self.year_lst = [str(y) for y in range(2015, 2016)]
    elif split == 'valid':
      self.year_lst = [str(y) for y in range(2023, 2024)]
    elif split == 'test':
      self.year_lst = [str(y) for y in range(2024, 2025)]
    else:
      raise ValueError(f"Unsupported dataset split: {split}")
    # self.month_lst = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    self.month_lst = ["12"]

    # other params
    # self.n_samples = 0
    # self.n_files = 0
    # for year in self.year_lst:
    #   for month in self.month_lst:
    #     reanalysis_file = os.path.join(self.era5_root, f"single_level/reanalysis/{year}/{month}/{variable}.nc")
    #     reanalysis_dataset = xr.open_dataset(reanalysis_file)
    #     assert len(reanalysis_dataset.data_vars) == 1
    #     data_array = reanalysis_dataset[list(reanalysis_dataset.data_vars)[0]].values
    #     self.n_sample += data_array.shape[0]
    #     self.n_files += 1

    # states
    self._start_new_epoch() # reset states
    self._load_next_file() # load data
  
  def _start_new_epoch(self):
    self.remaining_files = list(itertools.product(self.year_lst, self.month_lst))
    if self.split == 'train':
      random.shuffle(self.remaining_files)
    # states
    self.current_file = None
    self.current_data = None
    self.current_normalize_min = None
    self.current_normalize_max = None
    self.n_samples = None
    self.access_indices = None
    self.current_start_idx = None

  def _load_next_file(self):
    is_epoch_finished = False
    if not self.remaining_files:
      self._start_new_epoch()
      is_epoch_finished = True

    self.current_file = self.remaining_files.pop()
    year, month = self.current_file
    reanalysis_file = os.path.join(self.era5_root, f"single_level/reanalysis/{year}/{month}/{self.variable}.nc")
    reanalysis_dataset = xr.open_dataset(reanalysis_file)
    assert len(reanalysis_dataset.data_vars) == 1
    self.current_data = reanalysis_dataset[list(reanalysis_dataset.data_vars)[0]].values
    self.n_samples = self.current_data.shape[0]
    # normalize
    self.current_normalize_min = np.min(self.current_data, axis=(-1, -2), keepdims=True)
    self.current_normalize_max = np.max(self.current_data, axis=(-1, -2), keepdims=True)
    assert (self.current_normalize_min != self.current_normalize_max).all()
    self.current_data = (self.current_data - self.current_normalize_min) / (self.current_normalize_max - self.current_normalize_min)

    if self.split == 'train':
      self.access_indices = np.random.permutation(self.n_samples)
    else:
      self.access_indices = np.arange(self.n_samples)
    self.current_start_idx = 0

    return is_epoch_finished

  def _sample_batch(self, batch_size):
    # select batch
    n_remaining = self.n_samples - self.current_start_idx
    n_take = min(n_remaining, batch_size)
    selected_idx = self.access_indices[self.current_start_idx : self.current_start_idx + n_take]
    self.current_start_idx = self.current_start_idx + n_take

    # select patch
    if self.split == 'train':
      h = self.current_data.shape[-2]
      w = self.current_data.shape[-1]
      assert h >= self.patch_size
      assert w >= self.patch_size
      h_start_idx = np.random.randint(0, h - self.patch_size + 1, size = n_take)
      w_start_idx = np.random.randint(0, w - self.patch_size + 1, size = n_take)

      # idx 2D shape [n_take, patch_size]
      h_idx = h_start_idx[:, np.newaxis] + np.arange(self.patch_size)[np.newaxis, :]
      w_idx = w_start_idx[:, np.newaxis] + np.arange(self.patch_size)[np.newaxis, :]

      selected_data = self.current_data[selected_idx[:, np.newaxis, np.newaxis], h_idx[:, :, np.newaxis], w_idx[:, np.newaxis, :]]
    else:
      selected_data = self.current_data[selected_idx]

    selected_data_min = self.current_normalize_min[selected_idx]
    selected_data_max = self.current_normalize_max[selected_idx]

    if self.current_start_idx == self.n_samples:
      is_file_finished = True
    else:
      is_file_finished = False

    return selected_data, selected_data_min, selected_data_max, is_file_finished

  def sample_batch(self):
    batch_data = []
    normalize_min = []
    normalize_max = []
    current_batch_size = 0
    is_epoch_finished = False
    while current_batch_size < self.batch_size:
      selected_data, selected_data_min, selected_data_max, is_file_finished = self._sample_batch(self.batch_size)
      batch_size = len(selected_data)

      batch_data.append(selected_data)
      normalize_min.append(selected_data_min)
      normalize_max.append(selected_data_max)
      current_batch_size += batch_size

      # need more data for current batch
      if is_file_finished:
        is_epoch_finished = self._load_next_file()
        if is_epoch_finished:
          break

    batch_data_array = np.concatenate(batch_data, axis=0)  # shape [b, h, w]
    normalize_min_array = np.concatenate(normalize_min, axis=0)  # shape [b, h, w]
    normalize_max_array = np.concatenate(normalize_max, axis=0)  # shape [b, h, w]
    is_full_batch = current_batch_size == self.batch_size
    return batch_data_array, normalize_min_array, normalize_max_array, is_full_batch, is_epoch_finished

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

def crop(x, padding):
  return F.pad(
    x,
    (-padding[0], -padding[1], -padding[2], -padding[3]),
  )

def compute_total_bytes(out_enc):
  total_bytes = 0
  for out_strs in out_enc["strings"]:
    for out_str in out_strs:
      assert isinstance(out_str, bytes)
      total_bytes += len(out_str)
  return total_bytes

def eval_tcm_era5(configs):
  args = configs
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
  torch.use_deterministic_algorithms(True)
  current_proc_name = mp.current_process().name
  try:
    worker_idx = int(current_proc_name.split('-')[-1]) % configs.num_gpus
  except:
    worker_idx = 0
  p = 128

  # datasets
  variable = args.variable
  valid_dataset = Era5ReanalysisDataset(variable=variable, batch_size=1, split='valid')
  valid_dataloader = valid_dataset

  if args.cuda:
    device = f'cuda:{worker_idx}'
  else:
    device = 'cpu'
  net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=int(configs.n), M=320)
  net = net.to(device)
  net.eval()
  count = 0
  PSNR = 0
  Bit_rate = 0
  MS_SSIM = 0
  compression_ratio = 0
  total_time = 0
  dictory = {}
  if args.checkpoint:  # load from previous checkpoint
    print("Loading", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)
  if args.real:
    net.update()
    while True:
      batch_data_array, normalize_min_array, normalize_max_array, is_full_batch, is_epoch_finished = valid_dataloader.sample_batch()
      x = torch.Tensor(batch_data_array).float().to(device)
      x = x.unsqueeze(1).repeat(1, 3, 1, 1)
      x_padded, padding = pad(x, p)
      count += 1
      with torch.no_grad():
        if args.cuda:
          torch.cuda.synchronize(device)
        s = time.time()
        out_enc = net.compress(x_padded)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
        if args.cuda:
          torch.cuda.synchronize(device)
        e = time.time()
        total_time += (e - s)
        out_dec["x_hat"] = out_dec["x_hat"].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
        # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
        # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
        compressed_total_bytes = compute_total_bytes(out_enc)
        Bit_rate += compressed_total_bytes * 8.0 / num_pixels
        compression_ratio += num_pixels * 3 / compressed_total_bytes
        PSNR += compute_psnr(x, out_dec["x_hat"])
        MS_SSIM += compute_msssim(x, out_dec["x_hat"])
        # '''
        # Visualization
        # '''
        # image_name = os.path.basename(img_path)
        # if image_name in ["kodim04.png", "kodim12.png"]:
        #   error = torch.abs(x - out_dec["x_hat"])
        #   hat_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze(0).cpu())
        #   error_img = transforms.ToPILImage()((error*10).clamp(0,1).squeeze(0).cpu())
        #   Image.open(img_path).convert('RGB').save(f"./results/{image_name}")
        #   hat_img.save(f"./results/hat_{image_name}")
        #   error_img.save(f"./results/error_{image_name}")
      if is_epoch_finished:
        break

  else:
    while True:
      batch_data_array, normalize_min_array, normalize_max_array, is_full_batch, is_epoch_finished = valid_dataloader.sample_batch()
      x = torch.Tensor(batch_data_array).float().to(device)
      x = x.unsqueeze(1).repeat(1, 3, 1, 1)
      x_padded, padding = pad(x, p)
      count += 1
      with torch.no_grad():
        if args.cuda:
          torch.cuda.synchronize(device)
        s = time.time()
        out_net = net.forward(x_padded)
        if args.cuda:
          torch.cuda.synchronize(device)
        e = time.time()
        total_time += (e - s)
        out_net['x_hat'].clamp_(0, 1)
        out_net["x_hat"] = out_net["x_hat"].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        out_net["x_hat"] = crop(out_net["x_hat"], padding)
        # print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
        # print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
        # print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
        PSNR += compute_psnr(x, out_net["x_hat"])
        MS_SSIM += compute_msssim(x, out_net["x_hat"])
        Bit_rate += compute_bpp(out_net)
      if is_epoch_finished:
        break
  PSNR = PSNR / count
  MS_SSIM = MS_SSIM / count
  Bit_rate = Bit_rate / count
  total_time = total_time / count
  # print(f'Avg PSNR: {PSNR:.2f}dB')
  # print(f'Avg MS-SSIM: {MS_SSIM:.4f}')
  # print(f'Avg Bit-rate: {Bit_rate:.3f} bpp')
  # print(f'Avg compress-decompress time: {total_time:.3f} ms')
  result = {
    "worker_idx": worker_idx,
    "psnr": PSNR,
    "ms_ssim": MS_SSIM,
    "bit_rate": Bit_rate,
    "total_time": total_time,
  }
  if args.real:
    compression_ratio = compression_ratio / count
    result["compression_ratio"] = compression_ratio
    # print(f'Avg compression ratio: {compression_ratio:.2f}')
  return result
  
def main():
  print(f"{torch.cuda.is_available()=}")
  print(f"{torch.cuda.device_count()=}")
  '''
  params
  '''
  variable_lst = [
    "10m_u_component_of_wind",
    # "10m_v_component_of_wind",
    # "2m_temperature",
    # "total_precipitation"
  ]

  n_lst = ["64", "128"]
  lambda_lst = ["0.0025", "0.0035", "0.0067", "0.013", "0.025", "0.05"]
  # lambda_lst = ["0.05"]
  label_lst = [str(5*i) for i in range(11)] + ["best", "latest"]

  '''
  run eval
  '''
  num_gpus = torch.cuda.device_count()
  # num_gpus = 1
  ctx = mp.get_context('spawn')
  pool = ctx.Pool(processes=num_gpus)
  results = []
  for variable in variable_lst:
    for n in n_lst:
      for _lambda in lambda_lst:
        for label in label_lst:
          try:
            epoch = int(label)
            ckpt_name = f"{label}_checkpoint.pth.tar"
          except:
            ckpt_name = f"checkpoint_{label}.pth.tar"
          checkpoint = f"/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints/N_{n}_lambda_{_lambda}/{ckpt_name}"
          if os.path.exists(checkpoint):
            checkpoint_content = torch.load(checkpoint, map_location='cpu')
            epoch = checkpoint_content["epoch"] + 1
            print(f"[INFO] Starting TCM eval on {variable} with checkpoint {checkpoint}, epoch {epoch} ......")
            eval_configs = Config({
              "cuda" : True,
              "num_gpus": num_gpus,
              "n": n,
              "lambda": _lambda,
              "label": label,
              "epoch": epoch,
              "checkpoint": checkpoint,
              "variable": variable,
              "real": True,
            })
            if num_gpus > 1:
              eval_metrics = pool.apply_async(eval_tcm_era5, args = (eval_configs,))
            else:
              eval_metrics = eval_tcm_era5(eval_configs)
            result = {
              "variable": variable,
              "checkpoint": checkpoint,
              "n": n,
              "lambda": _lambda,
              "label": label,
              "epoch": epoch,
              "eval_metrics": eval_metrics,
            }
            results.append(result)
  
  pool.close()
  # pool.join()

  for idx in range(len(results)):
    async_result = results[idx]
    print(f"[INFO] Waiting TCM eval on {async_result['variable']} with checkpoint {async_result['checkpoint']}, epoch {async_result['epoch']} ......")
    if num_gpus > 1:
      result = {
        "variable": async_result["variable"],
        "checkpoint": async_result["checkpoint"],
        "n": async_result["n"],
        "lambda": async_result["lambda"],
        "label": async_result["label"],
        "epoch": async_result["epoch"],
        **(async_result["eval_metrics"].get()),
      }
    else:
      result = {
        "variable": async_result["variable"],
        "checkpoint": async_result["checkpoint"],
        "n": async_result["n"],
        "lambda": async_result["lambda"],
        "label": async_result["label"],
        "epoch": async_result["epoch"],
        **(async_result["eval_metrics"]),
      }
    
    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    df.to_csv("./results/eval_tcm_ckpt_era5.csv", index=False)
  
  pool.join()


if __name__ == "__main__":
  main()
  