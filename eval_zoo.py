import torch
import torch.nn.functional as F
from torchvision import transforms
from compressai.zoo import image_models
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

def compute_total_bytes(out_enc):
  total_bytes = 0
  for out_strs in out_enc["strings"]:
    for out_str in out_strs:
      assert isinstance(out_str, bytes)
      total_bytes += len(out_str)
  return total_bytes

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

def eval_zoo(configs):
  args = configs
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
  torch.use_deterministic_algorithms(True)
  current_proc_name = mp.current_process().name
  try:
    worker_idx = int(current_proc_name.split('-')[-1]) % configs.num_gpus
  except:
    worker_idx = 0
  p = 128
  path = args.data
  img_list = []
  for file in os.listdir(path):
    if file[-3:] in ["jpg", "png", "peg"]:
      img_list.append(file)
  if args.cuda:
    device = f'cuda:{worker_idx}'
  else:
    device = 'cpu'
  net = image_models[args.model](quality=args.quality_factor, pretrained=True)
  net = net.to(device)
  net.eval()
  count = 0
  PSNR = 0
  Bit_rate = 0
  MS_SSIM = 0
  compression_ratio = 0
  total_time = 0
  if args.real:
    net.update()
    for img_name in img_list:
      img_path = os.path.join(path, img_name)
      img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
      x = img.unsqueeze(0)
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
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
        # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
        # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
        compressed_total_bytes = compute_total_bytes(out_enc)
        Bit_rate += compressed_total_bytes * 8.0 / num_pixels
        compression_ratio += num_pixels * 3 / compressed_total_bytes
        PSNR += compute_psnr(x, out_dec["x_hat"])
        MS_SSIM += compute_msssim(x, out_dec["x_hat"])

  else:
    for img_name in img_list:
      img_path = os.path.join(path, img_name)
      img = Image.open(img_path).convert('RGB')
      x = transforms.ToTensor()(img).unsqueeze(0).to(device)
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
        # print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
        # print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
        # print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
        PSNR += compute_psnr(x, out_net["x_hat"])
        MS_SSIM += compute_msssim(x, out_net["x_hat"])
        Bit_rate += compute_bpp(out_net)
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
  datasets = {
    "Kodak": "/capstor/scratch/cscs/ljiayong/datasets/kodak-kaggle",
    "CLIC": "/capstor/scratch/cscs/ljiayong/datasets/CLIC_2021/test",
  }
  models = {
    "Cheng(CVPR20)": "cheng2020-anchor",
    "Minnen(NeurIPS18)": "mbt2018",
    "Balle(ICLR18)": "bmshj2018-hyperprior",
  }
  quality_factors = {
    "Cheng(CVPR20)": [1, 2, 3, 4, 5, 6],
    "Minnen(NeurIPS18)": [1, 2, 3, 4, 5, 6, 7, 8],
    "Balle(ICLR18)": [1, 2, 3, 4, 5, 6, 7, 8],
  }

  '''
  run eval
  '''
  num_gpus = torch.cuda.device_count()
  # num_gpus = 1
  ctx = mp.get_context('spawn')
  pool = ctx.Pool(processes=num_gpus)
  results = []
  for dataset_name in datasets:
    for model_name in models:
      for quality_factor in quality_factors[model_name]:
        print(f"[INFO] Starting {model_name} eval on {dataset_name} dataset with quality factor {quality_factor} ......")
        eval_configs = Config({
          "model": models[model_name],
          "quality_factor": quality_factor,
          "cuda" : True,
          "num_gpus": num_gpus,
          "data": datasets[dataset_name],
          "real": True,
        })
        if num_gpus > 1:
          eval_metrics = pool.apply_async(eval_zoo, args = (eval_configs,))
        else:
          eval_metrics = eval_zoo(eval_configs)
        result = {
          "dataset_name": dataset_name,
          "model_name": model_name,
          "quality_factor": quality_factor,
          "eval_metrics": eval_metrics,
        }
        results.append(result)

  pool.close()
  # pool.join()

  for idx in range(len(results)):
    async_result = results[idx]
    print(f"[INFO] Waiting {async_result['model_name']} eval on {async_result['dataset_name']} dataset with quality factor {async_result['quality_factor']} ......")
    if num_gpus > 1:
      result = {
        "dataset_name": async_result["dataset_name"],
        "model_name": async_result["model_name"],
        "quality_factor": async_result["quality_factor"],
        **(async_result["eval_metrics"].get()),
      }
    else:
      result = {
        "dataset_name": async_result["dataset_name"],
        "model_name": async_result["model_name"],
        "quality_factor": async_result["quality_factor"],
        **(async_result["eval_metrics"]),
      }
    
    results[idx] = result
    df = pd.DataFrame(results[:idx+1])
    df.to_csv("./results/eval_zoo.csv", index=False)
  
  pool.join()


if __name__ == "__main__":
  main()
  