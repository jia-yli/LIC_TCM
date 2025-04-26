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
from tqdm import tqdm
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

def eval_tcm(configs):
  args = configs
  p = 128
  path = args.data
  img_list = []
  for file in os.listdir(path):
    if file[-3:] in ["jpg", "png", "peg"]:
      img_list.append(file)
  if args.cuda:
    device = 'cuda:0'
  else:
    device = 'cpu'
  net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
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
    for img_name in tqdm(img_list):
      img_path = os.path.join(path, img_name)
      img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
      x = img.unsqueeze(0)
      x_padded, padding = pad(x, p)
      count += 1
      with torch.no_grad():
        if args.cuda:
          torch.cuda.synchronize()
        s = time.time()
        out_enc = net.compress(x_padded)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
        if args.cuda:
          torch.cuda.synchronize()
        e = time.time()
        total_time += (e - s)
        out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
        # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
        # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
        Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        compression_ratio += num_pixels * 3 / sum(len(s[0]) for s in out_enc["strings"])
        PSNR += compute_psnr(x, out_dec["x_hat"])
        MS_SSIM += compute_msssim(x, out_dec["x_hat"])

  else:
    for img_name in tqdm(img_list):
      img_path = os.path.join(path, img_name)
      img = Image.open(img_path).convert('RGB')
      x = transforms.ToTensor()(img).unsqueeze(0).to(device)
      x_padded, padding = pad(x, p)
      count += 1
      with torch.no_grad():
        if args.cuda:
          torch.cuda.synchronize()
        s = time.time()
        out_net = net.forward(x_padded)
        if args.cuda:
          torch.cuda.synchronize()
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
  print(f'Avg PSNR: {PSNR:.2f}dB')
  print(f'Avg MS-SSIM: {MS_SSIM:.4f}')
  print(f'Avg Bit-rate: {Bit_rate:.3f} bpp')
  print(f'Avg compress-decompress time: {total_time:.3f} ms')
  result = {
    "psnr": PSNR,
    "ms_ssim": MS_SSIM,
    "bit_rate": Bit_rate,
    "total_time": total_time,
  }
  if args.real:
    compression_ratio = compression_ratio / count
    result["compression_ratio"] = compression_ratio
    print(f'Avg compression ratio: {compression_ratio:.2f}')
  return result
  
def main():
  print(f"{torch.cuda.is_available()=}")
  '''
  params
  '''
  datasets = {
    "Kodak": "/capstor/scratch/cscs/ljiayong/datasets/kodak-kaggle",
    "CLIC": "/capstor/scratch/cscs/ljiayong/datasets/CLIC_2021/test",
  }
  checkpoint_lst = [
    "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.0025.pth.tar",
    "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.0035.pth.tar",
    "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.0067.pth.tar",
    "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.013.pth.tar",
    "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.025.pth.tar",
    "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.05.pth.tar",
    # "/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar",
  ]

  '''
  run eval
  '''
  results = []
  for dataset_name in datasets:
    for checkpoint in checkpoint_lst:
      print(f"[INFO] Starting TCM eval on {dataset_name} dataset with checkpoint {checkpoint} ......")
      eval_configs = Config({
        "cuda" : True,
        "checkpoint": checkpoint,
        "data": datasets[dataset_name],
        "real": True,
      })
      eval_metrics = eval_tcm(eval_configs)
      result = {
        "dataset_name": dataset_name,
        "checkpoint": checkpoint,
        **eval_metrics
      }
      results.append(result)
  
  df = pd.DataFrame(results)
  df.to_csv("./results/eval_tcm.csv", index=False)


if __name__ == "__main__":
  main()
  