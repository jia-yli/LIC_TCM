import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM, TCMWeighted
from torch.utils.tensorboard import SummaryWriter   
import os

import numpy as np
import pandas as pd

from datetime import datetime, timezone
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
  return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
  """Custom rate distortion loss with a Lagrangian parameter."""

  def __init__(self, lmbda=1e-2):
    super().__init__()
    self.lmbda = lmbda

  def forward(self, output, target, weight):
    assert (output["x_hat"].shape == target.shape == weight.shape)
    N, _, H, W = target.size()
    out = {}
    num_pixels = N * H * W

    out["bpp_loss"] = sum(
      (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
      for likelihoods in output["likelihoods"].values()
    )

    out["mse_loss"] = ((output["x_hat"] - target).pow(2) / (weight.pow(2))).mean()
    out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

    return out

# def normalize_weight_arr(weight, thr=10):
#   # [N, C, H, W]
#   rms = np.sqrt(np.mean(weight**2, axis=(1,2,3), keepdims=True))
#   weight_norm = np.clip(weight / rms, 1/thr, thr)
#   weight01 = np.log(weight_norm) / np.log(thr) # [-1, 1]
#   weight01 = (weight01 + 1) / 2 # [0, 1]
#   return weight_norm, weight01

def normalize_weight_tensor(weight, thr, use_log_norm_weight):
  # [N, C, H, W]
  if use_log_norm_weight:
    log_avg = torch.exp(torch.log(weight + 1e-6).mean(dim=(1, 2, 3), keepdim=True))
    weight_norm = torch.clamp(weight / log_avg, 1/thr, thr)
    weight01 = torch.log(weight_norm) / np.log(thr)
  else:
    rms = torch.sqrt(torch.mean(weight ** 2, dim=(1, 2, 3), keepdim=True))
    weight_norm = torch.clamp(weight / rms, 1/thr, thr)
    weight01 = torch.log(weight_norm) / np.log(thr)
  return weight_norm, weight01


class AverageMeter:
  """Compute running average."""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
  """Custom DataParallel to access the module methods."""

  def __getattr__(self, key):
    try:
      return super().__getattr__(key)
    except AttributeError:
      return getattr(self.module, key)


def configure_optimizers(net, args):
  """Separate parameters for the main optimizer and the auxiliary optimizer.
  Return two optimizers"""

  parameters = {
    n
    for n, p in net.named_parameters()
    if not n.endswith(".quantiles") and p.requires_grad
  }
  aux_parameters = {
    n
    for n, p in net.named_parameters()
    if n.endswith(".quantiles") and p.requires_grad
  }

  # Make sure we don't have an intersection of parameters
  params_dict = dict(net.named_parameters())
  inter_params = parameters & aux_parameters
  union_params = parameters | aux_parameters

  assert len(inter_params) == 0
  assert len(union_params) - len(params_dict.keys()) == 0

  optimizer = optim.Adam(
    (params_dict[n] for n in sorted(parameters)),
    lr=args.learning_rate,
  )
  aux_optimizer = optim.Adam(
    (params_dict[n] for n in sorted(aux_parameters)),
    lr=args.aux_learning_rate,
  )
  return optimizer, aux_optimizer


def train_one_epoch(
  model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, weighted_ratio, freeze_aux_loss, use_gray_scale_weight, use_log_norm_weight
):
  model.train()
  device = next(model.parameters()).device

  for i, d in enumerate(train_dataloader):
    batch_size = len(d)//2
    x = d[:batch_size].to(device)

    b = d[-batch_size:].to(device)
    if use_gray_scale_weight:
      b = b * 0 + b.mean(dim=1, keepdim=True)
    plain_mask = torch.rand(batch_size, device=b.device) >= weighted_ratio
    b[plain_mask, ...] = 1

    b_norm, b01 = normalize_weight_tensor(b, 10, use_log_norm_weight)
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    out_net = model(x, b01)

    out_criterion = criterion(out_net, x, b_norm)
    out_criterion["loss"].backward()
    if clip_max_norm > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    optimizer.step()

    aux_loss = model.aux_loss()
    if not freeze_aux_loss:
      aux_loss.backward()
      aux_optimizer.step()

    if i % 100 == 0:
      print(
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} "
        f"Lambda {criterion.lmbda} Train epoch {epoch}: ["
        f"{i*len(d)} Samples]"
        f' Loss: {out_criterion["loss"].item()} |'
        f' MSE loss: {out_criterion["mse_loss"].item()} |'
        f' Bpp loss: {out_criterion["bpp_loss"].item()} |'
        f" Aux loss: {aux_loss.item()}"
      )


def valid_epoch(epoch, valid_dataloader, model, criterion, mode, use_gray_scale_weight, use_log_norm_weight):
  model.eval()
  device = next(model.parameters()).device

  loss = AverageMeter()
  bpp_loss = AverageMeter()
  mse_loss = AverageMeter()
  aux_loss = AverageMeter()

  with torch.no_grad():
    for d in valid_dataloader:
      batch_size = len(d)//2
      x = d[:batch_size].to(device)
      b = d[-batch_size:].to(device)
      if use_gray_scale_weight:
        b = b * 0 + b.mean(dim=1, keepdim=True)
      if mode == 'plain':
        b_norm, b01 = normalize_weight_tensor(b * 0 + 1, 10, use_log_norm_weight)
      elif mode == "weighted":
        b_norm, b01 = normalize_weight_tensor(b, 10, use_log_norm_weight)
      else:
        raise ValueError(f"Unsupported mode {mode}")

      out_net = model(x, b01)
      out_criterion = criterion(out_net, x, b_norm)

      aux_loss.update(model.aux_loss())
      bpp_loss.update(out_criterion["bpp_loss"])
      loss.update(out_criterion["loss"])
      mse_loss.update(out_criterion["mse_loss"])

  result = {
    "epoch": epoch,
    "mode": mode,
    "loss": loss.avg.item(),
    "mse": mse_loss.avg.item(),
    "bpp": bpp_loss.avg.item(),
    "aux": aux_loss.avg.item()
  }

  print(
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} "
    f"Lambda {criterion.lmbda} Valid epoch {epoch} mode {mode}: Average losses:"
    f" Loss: {loss.avg} |"
    f" MSE loss: {mse_loss.avg} |"
    f" Bpp loss: {bpp_loss.avg} |"
    f" Aux loss: {aux_loss.avg}"
  )

  return result


def save_checkpoint(state, is_best, epoch, save_path, filename):
  torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
  if epoch % 3 == 0:
    torch.save(state, filename)
  if is_best:
    torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))


def parse_args(argv):
  parser = argparse.ArgumentParser(description="Example training script.")
  parser.add_argument(
    "-d", "--dataset", type=str, required=True, help="Training dataset"
  )
  parser.add_argument(
    "-e",
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (default: %(default)s)",
  )
  parser.add_argument(
    "-lr",
    "--learning-rate",
    default=1e-4,
    type=float,
    help="Learning rate (default: %(default)s)",
  )
  parser.add_argument(
    "-n",
    "--num-workers",
    type=int,
    default=16,
    help="Dataloaders threads (default: %(default)s)",
  )
  parser.add_argument(
    "--N", type=int, default=128,
  )
  parser.add_argument(
    "--lambda",
    dest="lmbda",
    type=float,
    default=3,
    help="Bit-rate distortion parameter (default: %(default)s)",
  )
  parser.add_argument(
    "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
  )
  parser.add_argument(
    "--valid-batch-size",
    type=int,
    default=8,
    help="Valid batch size (default: %(default)s)",
  )
  parser.add_argument(
    "--aux-learning-rate",
    default=1e-3,
    help="Auxiliary loss learning rate (default: %(default)s)",
  )
  parser.add_argument(
    "--patch-size",
    type=int,
    default=256,
    help="Size of the patches to be cropped (default: %(default)s)",
  )
  parser.add_argument(
    "--seed", type=float, default=42, help="Set random seed for reproducibility"
  )
  parser.add_argument(
    "--clip_max_norm",
    default=0.2,
    type=float,
    help="gradient clipping max norm (default: %(default)s",
  )
  parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
  parser.add_argument("--save-path", type=str, help="save_path")
  parser.add_argument(
    "--lr_epoch", nargs='+', type=int
  )
  parser.add_argument(
    "--continue-train", action="store_true"
  )
  parser.add_argument(
    "--freeze-pretrained", action="store_true"
  )
  parser.add_argument(
    "--weighted-ratio", type=float, default=0.5
  )
  parser.add_argument(
    "--use-gray-scale-weight", type=int, default=0
  )
  parser.add_argument(
    "--use-log-norm-weight", type=int, default=0
  )
  parser.add_argument(
    "--use-weight-in-decoder", type=int, default=0
  )
  args = parser.parse_args(argv)
  return args


def main(argv):
  args = parse_args(argv)
  for arg in vars(args):
    print(arg, ":", getattr(args, arg))
  save_path = os.path.join(args.save_path, f"N_{args.N}_lambda_{str(args.lmbda)}")
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, "tensorboard"))
  if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
  writer = SummaryWriter(os.path.join(save_path, "tensorboard"))

  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
  torch.use_deterministic_algorithms(True)
  # datasets
  train_transforms = transforms.Compose(
    [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
  )

  valid_transforms = transforms.Compose(
    [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
  )

  train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
  valid_dataset = ImageFolder(args.dataset, split="valid", transform=valid_transforms)

  device = 'cuda:0'

  train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size*2,
    num_workers=args.num_workers,
    shuffle=True,
    pin_memory=True,
  )

  valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=args.valid_batch_size*2,
    num_workers=args.num_workers,
    shuffle=False,
    pin_memory=True,
  )

  # net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
  net = TCMWeighted(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320, use_weight_in_decoder=args.use_weight_in_decoder)
  if args.freeze_pretrained:
    print("[INFO] Pretrained modules are freezed")
    net.freeze_pretrained_modules = True
  net = net.to(device)

  optimizer, aux_optimizer = configure_optimizers(net, args)
  milestones = args.lr_epoch
  print("milestones: ", milestones)
  lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

  criterion = RateDistortionLoss(lmbda=args.lmbda)

  last_epoch = 0
  if args.checkpoint:  # load from previous checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # handle DP
    dictory = {}
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)

    if args.continue_train:
      last_epoch = checkpoint["epoch"] + 1
      optimizer.load_state_dict(checkpoint["optimizer"])
      aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
      lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    else:
      save_checkpoint(
        {
          "epoch": -1,
          "state_dict": net.state_dict(),
        },
        False,
        -1,
        save_path,
        os.path.join(save_path, f"-1_checkpoint.pth.tar"),
      )
    print(f"Loaded {args.checkpoint} and continue train from epoch {last_epoch}, ({args.continue_train=})")

  if torch.cuda.device_count() > 1:
    net = CustomDataParallel(net)
    print(f'[INFO] DataParallel: Using {torch.cuda.device_count()} GPUs') 

  best_loss = float("inf")
  results = []
  for epoch in range(last_epoch, args.epochs):
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    train_one_epoch(
      net,
      criterion,
      train_dataloader,
      optimizer,
      aux_optimizer,
      epoch,
      args.clip_max_norm,
      args.weighted_ratio,
      freeze_aux_loss = args.freeze_pretrained and not args.use_weight_in_decoder,
      use_gray_scale_weight = args.use_gray_scale_weight,
      use_log_norm_weight = args.use_log_norm_weight
    )
    valid_result_weighted = valid_epoch(epoch, valid_dataloader, net, criterion, mode="weighted", use_gray_scale_weight=args.use_gray_scale_weight, use_log_norm_weight=args.use_log_norm_weight)
    valid_result_plain = valid_epoch(epoch, valid_dataloader, net, criterion, mode="plain", use_gray_scale_weight=args.use_gray_scale_weight, use_log_norm_weight=args.use_log_norm_weight)
    results.append(valid_result_weighted)
    results.append(valid_result_plain)
    loss = valid_result_weighted["loss"]
    writer.add_scalar('valid_loss', loss, epoch)
    lr_scheduler.step()

    is_best = loss < best_loss
    best_loss = min(loss, best_loss)

    pd.DataFrame(results).to_csv(os.path.join(save_path, "results.csv"), index=False)

    save_checkpoint(
      {
        "epoch": epoch,
        "state_dict": net.state_dict(),
        "loss": loss,
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
      },
      is_best,
      epoch,
      save_path,
      os.path.join(save_path, f"{epoch}_checkpoint.pth.tar"),
    )


if __name__ == "__main__":
  main(sys.argv[1:])