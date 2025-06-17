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

from models import TCM
from torch.utils.tensorboard import SummaryWriter   
import os
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import itertools

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

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
    self.month_lst = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

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
    if (self.split == 'train') and (self.patch_size > 0):
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

def compute_msssim(a, b):
  return ms_ssim(a, b, data_range=1.)

class RateDistortionLoss(nn.Module):
  """Custom rate distortion loss with a Lagrangian parameter."""

  def __init__(self, lmbda=1e-2, type='mse'):
    super().__init__()
    self.mse = nn.MSELoss()
    self.lmbda = lmbda
    self.type = type

  def forward(self, output, target):
    N, _, H, W = target.size()
    out = {}
    num_pixels = N * H * W

    out["bpp_loss"] = sum(
      (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
      for likelihoods in output["likelihoods"].values()
    )
    if self.type == 'mse':
      out["mse_loss"] = self.mse(output["x_hat"], target)
      out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
    else:
      out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
      out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

    return out


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

def train_one_epoch(
  model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse'
):
  model.train()
  device = next(model.parameters()).device

  # for i, d in enumerate(train_dataloader):
  i = 0
  while True:
    batch_data_array, normalize_min_array, normalize_max_array, is_full_batch, is_epoch_finished = train_dataloader.sample_batch()
    if not is_full_batch:
      assert is_epoch_finished
      break
    d = torch.Tensor(batch_data_array).float().to(device)
    d = d.unsqueeze(1).repeat(1, 3, 1, 1)
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    d_padded, padding = pad(d, 128)
    out_net = model(d_padded)
    out_net["x_hat"] = out_net["x_hat"].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    out_net["x_hat"] = crop(out_net["x_hat"], padding)

    out_criterion = criterion(out_net, d)
    out_criterion["loss"].backward()
    if clip_max_norm > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    optimizer.step()

    aux_loss = model.aux_loss()
    aux_loss.backward()
    aux_optimizer.step()

    if i % 100 == 0:
      if type == 'mse':
        print(
          f"Lambda {criterion.lmbda} Train epoch {epoch}: ["
          f"{i*len(d)} Samples]"
          f'\tLoss: {out_criterion["loss"].item():.3f} |'
          f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
          f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
          f"\tAux loss: {aux_loss.item():.2f}"
        )
      else:
        print(
          f"Lambda {criterion.lmbda} Train epoch {epoch}: ["
          f"{i*len(d)} Samples]"
          f'\tLoss: {out_criterion["loss"].item():.3f} |'
          f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
          f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
          f"\tAux loss: {aux_loss.item():.2f}"
        )
    i += 1
    if is_epoch_finished:
      break

def valid_epoch(epoch, valid_dataloader, model, criterion, type='mse'):
  model.eval()
  device = next(model.parameters()).device
  if type == 'mse':
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
      while True:
        batch_data_array, normalize_min_array, normalize_max_array, is_full_batch, is_epoch_finished = valid_dataloader.sample_batch()
        d = torch.Tensor(batch_data_array).float().to(device)
        d = d.unsqueeze(1).repeat(1, 3, 1, 1)
        d_padded, padding = pad(d, 128)
        out_net = model(d_padded)
        out_net["x_hat"] = out_net["x_hat"].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        out_net["x_hat"] = crop(out_net["x_hat"], padding)
        out_criterion = criterion(out_net, d)

        aux_loss.update(model.aux_loss())
        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        if is_epoch_finished:
          break

    print(
      f"Valid epoch {epoch}: Average losses:"
      f"\tLoss: {loss.avg:.3f} |"
      f"\tMSE loss: {mse_loss.avg:.3f} |"
      f"\tBpp loss: {bpp_loss.avg:.2f} |"
      f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

  else:
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
      while True:
        batch_data_array, normalize_min_array, normalize_max_array, is_full_batch, is_epoch_finished = valid_dataloader.sample_batch()
        d = torch.Tensor(batch_data_array).float().to(device)
        d = d.unsqueeze(1).repeat(1, 3, 1, 1)
        d_padded, padding = pad(d, 128)
        out_net = model(d_padded)
        out_net["x_hat"] = out_net["x_hat"].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        out_net["x_hat"] = crop(out_net["x_hat"], padding)
        out_criterion = criterion(out_net, d)

        aux_loss.update(model.aux_loss())
        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
        if is_epoch_finished:
          break

    print(
      f"Valid epoch {epoch}: Average losses:"
      f"\tLoss: {loss.avg:.3f} |"
      f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
      f"\tBpp loss: {bpp_loss.avg:.2f} |"
      f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

  return loss.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
  torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
  # if epoch % 5 == 0:
  torch.save(state, filename)
  if is_best:
    torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))


def parse_args(argv):
  parser = argparse.ArgumentParser(description="Example training script.")
  parser.add_argument(
    "-m",
    "--model",
    default="bmshj2018-factorized",
    choices=models.keys(),
    help="Model architecture (default: %(default)s)",
  )
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
    default=20,
    help="Dataloaders threads (default: %(default)s)",
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
  parser.add_argument("--cuda", action="store_true", help="Use cuda")
  parser.add_argument(
    "--save", action="store_true", default=True, help="Save model to disk"
  )
  parser.add_argument(
    "--seed", type=float, default=100, help="Set random seed for reproducibility"
  )
  parser.add_argument(
    "--clip_max_norm",
    default=0.2,
    type=float,
    help="gradient clipping max norm (default: %(default)s",
  )
  parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
  parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
  parser.add_argument("--save_path", type=str, help="save_path")
  parser.add_argument(
    "--skip_epoch", type=int, default=0
  )
  parser.add_argument(
    "--N", type=int, default=128,
  )
  parser.add_argument(
    "--lr_epoch", nargs='+', type=int
  )
  parser.add_argument(
    "--continue-train", action="store_true"
  )
  args = parser.parse_args(argv)
  return args


def main(argv):
  args = parse_args(argv)
  for arg in vars(args):
    print(arg, ":", getattr(args, arg))
  type = args.type
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
  variable = "10m_u_component_of_wind"
  train_dataset = Era5ReanalysisDataset(variable=variable, batch_size=args.batch_size, patch_size=args.patch_size, split='train')
  valid_dataset = Era5ReanalysisDataset(variable=variable, batch_size=8, split='valid')


  device = 'cuda:0'

  train_dataloader = train_dataset

  valid_dataloader = valid_dataset

  net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N, M=320)
  net = net.to(device)

  if args.cuda and torch.cuda.device_count() > 1:
    net = CustomDataParallel(net)

  optimizer, aux_optimizer = configure_optimizers(net, args)
  milestones = args.lr_epoch
  print("milestones: ", milestones)
  lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

  criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

  last_epoch = 0
  default_checkpoint_path = os.path.join(save_path, "checkpoint_latest.pth.tar")
  if args.checkpoint or os.path.exists(default_checkpoint_path):  # load from previous checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
      checkpoint_path = default_checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    if args.continue_train:
      last_epoch = checkpoint["epoch"] + 1
      optimizer.load_state_dict(checkpoint["optimizer"])
      aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
      lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    else:
      if args.save:
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
    print(f"Loaded {checkpoint_path} and continue train from epoch {last_epoch}, ({args.continue_train=})")

  best_loss = float("inf")
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
      type
    )
    loss = valid_epoch(epoch, valid_dataloader, net, criterion, type)
    writer.add_scalar('valid_loss', loss, epoch)
    lr_scheduler.step()

    is_best = loss < best_loss
    best_loss = min(loss, best_loss)

    if args.save:
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