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
import numpy as np
import h5py
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
  variable = args.variable
  era5_hdf5_path = f'/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/test'
  reanalysis_file_path = os.path.join(era5_hdf5_path, f'{variable}.hdf5')
  uncertainty_file_path = os.path.join(era5_hdf5_path, f'{variable}_interpolated_ensemble_spread.hdf5')
  with h5py.File(reanalysis_file_path, 'r') as hdf5_reanalysis:
    assert len(list(hdf5_reanalysis.keys())) == 1
    var_name = list(hdf5_reanalysis.keys())[0]
    reanalysis_data = np.array(hdf5_reanalysis[var_name])  # Read dataset, 1 month data: (744, 721, 1440)
  with h5py.File(uncertainty_file_path, 'r') as hdf5_uncertainty:
    uncertainty_data = np.array(hdf5_uncertainty[var_name])

  data_tensor = torch.Tensor(reanalysis_data).float()
  data_tensor = data_tensor.unsqueeze(1).repeat(1,3,1,1) # [N, C, H, W] with FP32 in Range [0, 1]
  uncertainty_tensor = torch.Tensor(uncertainty_data).float()
  uncertainty_tensor = uncertainty_tensor.unsqueeze(1).repeat(1,3,1,1) # [N, C, H, W] with FP32 in Range [0, 1]
  # min-max scaler
  min_val = data_tensor.amin(dim=(2, 3), keepdim=True)  # shape (N, C, 1, 1)
  max_val = data_tensor.amax(dim=(2, 3), keepdim=True)  # shape (N, C, 1, 1)

  diff = max_val - min_val
  assert (diff == 0).sum().item() == 0

  data_tensor = (data_tensor - min_val) / diff
  uncertainty_tensor = uncertainty_tensor / diff
  num_images = data_tensor.shape[0]
  error_bound_to_uncertainty_lst = [1, 0.5, 0.1]

  if args.cuda:
    device = f'cuda:{worker_idx}'
  else:
    device = 'cpu'
  net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=int(configs.n), M=320)
  net = net.to(device)
  net.eval()
  count = 0
  PSNR = 0
  PSNR_goals = [0] * len(error_bound_to_uncertainty_lst)
  Bit_rate = 0
  MS_SSIM = 0
  MS_SSIM_goals = [0] * len(error_bound_to_uncertainty_lst)
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
    for img_idx in range(num_images):
      x = data_tensor[img_idx:img_idx+1].to(device)
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
        out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
        out_dec["x_hat"] = out_dec["x_hat"] * 0 + out_dec["x_hat"].mean(dim=1, keepdim=True)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
        # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
        # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
        compressed_total_bytes = compute_total_bytes(out_enc)
        Bit_rate += compressed_total_bytes * 8.0 / num_pixels
        compression_ratio += num_pixels * 4 / compressed_total_bytes
        PSNR += compute_psnr(x, out_dec["x_hat"])
        MS_SSIM += compute_msssim(x, out_dec["x_hat"])
        for idx, error_bound_to_uncertainty in enumerate(error_bound_to_uncertainty_lst):
          uncertainty = uncertainty_tensor[img_idx:img_idx+1].to(device)
          PSNR_goals[idx] += compute_psnr(x, x + error_bound_to_uncertainty*uncertainty)
          MS_SSIM_goals[idx] += compute_msssim(x, x + error_bound_to_uncertainty*uncertainty)
        '''
        Visualization
        '''
        if img_idx in [0, 24]:
          error = torch.abs(x - out_dec["x_hat"])
          img = transforms.ToPILImage()(x.squeeze(0).cpu())
          hat_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze(0).cpu())
          error_img = transforms.ToPILImage()((error*10).clamp(0,1).squeeze(0).cpu())
          save_dir = f"./results/{args['n']}_{args['lambda']}_era5"
          os.makedirs(save_dir, exist_ok=True)
          img.save(os.path.join(save_dir, f"{args['variable']}_{img_idx}.png"))
          hat_img.save(os.path.join(save_dir, f"hat_{args['variable']}_{img_idx}.png"))
          error_img.save(os.path.join(save_dir, f"error_{args['variable']}_{img_idx}.png"))

  else:
    for img_idx in range(num_images):
      x = data_tensor[img_idx:img_idx+1].to(device)
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
        out_net["x_hat"] = crop(out_net["x_hat"], padding)
        out_dec["x_hat"] = out_dec["x_hat"] * 0 + out_dec["x_hat"].mean(dim=1, keepdim=True)
        # print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
        # print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
        # print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
        PSNR += compute_psnr(x, out_net["x_hat"])
        MS_SSIM += compute_msssim(x, out_net["x_hat"])
        Bit_rate += compute_bpp(out_net)
        for idx, error_bound_to_uncertainty in enumerate(error_bound_to_uncertainty_lst):
          uncertainty = uncertainty_tensor[img_idx:img_idx+1].to(device)
          PSNR_goals[idx] += compute_psnr(x, x + error_bound_to_uncertainty*uncertainty)
          MS_SSIM_goals[idx] += compute_msssim(x, x + error_bound_to_uncertainty*uncertainty)
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
  for idx, error_bound_to_uncertainty in enumerate(error_bound_to_uncertainty_lst):
    PSNR_goals[idx] = PSNR_goals[idx]/count
    MS_SSIM_goals[idx] = MS_SSIM_goals[idx]/count
    result[f'psnr_goal_{error_bound_to_uncertainty}'] = PSNR_goals[idx]
    result[f'ms_ssim_goal_{error_bound_to_uncertainty}'] = MS_SSIM_goals[idx]
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
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation"
  ]

  n_lst = ["64", "128"]
  lambda_lst = ["0.0025", "0.0035", "0.0067", "0.013", "0.025", "0.05"]

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
        checkpoint = f"/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_{n}_lambda_{_lambda}.pth.tar"
        if os.path.exists(checkpoint):
          print(f"[INFO] Starting TCM eval on {variable} with checkpoint {checkpoint} ......")
          eval_configs = Config({
            "cuda" : True,
            "num_gpus": num_gpus,
            "n": n,
            "lambda": _lambda,
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
            "eval_metrics": eval_metrics,
          }
          results.append(result)
  
  pool.close()
  # pool.join()

  for idx in range(len(results)):
    async_result = results[idx]
    print(f"[INFO] Waiting TCM eval on {async_result['variable']} with checkpoint {async_result['checkpoint']} ......")
    if num_gpus > 1:
      result = {
        "variable": async_result["variable"],
        "checkpoint": async_result["checkpoint"],
        "n": async_result["n"],
        "lambda": async_result["lambda"],
        **(async_result["eval_metrics"].get()),
      }
    else:
      result = {
        "variable": async_result["variable"],
        "checkpoint": async_result["checkpoint"],
        "n": async_result["n"],
        "lambda": async_result["lambda"],
        **(async_result["eval_metrics"]),
      }
    
    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    df.to_csv("./results/eval_tcm_era5.csv", index=False)
  
  pool.join()


if __name__ == "__main__":
  main()
  