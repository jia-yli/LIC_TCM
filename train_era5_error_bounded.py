import os
import math
import fire
import random
import inspect
import pickle
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

from datetime import datetime, timezone

from models import TCM, TCMWeighted 
from era5_dataset import Era5ReanalysisDatasetSingleLevel

class CustomDataParallel(nn.DataParallel):
  """Custom DataParallel to access the module methods."""

  def __getattr__(self, key):
    try:
      return super().__getattr__(key)
    except AttributeError:
      return getattr(self.module, key)

class ErrorBoundedRateOnlyLoss(nn.Module):
  """Custom rate distortion loss with a Lagrangian parameter."""

  def __init__(self, score_type, surrogate_loss_type, tau):
    super().__init__()
    self.score_type = score_type
    self.surrogate_loss_type = surrogate_loss_type
    self.tau = tau

  def forward(self, out, inp, target):
    assert (out["x_hat"].shape == inp.shape == target.shape)
    target = target * 0.99 # left some margin

    N, H, W = inp.shape
    num_pixels = N * H * W

    bpp_compression = sum(
      (torch.log(likelihoods).sum() / -math.log(2) / num_pixels)
      for likelihoods in out["likelihoods"].values()
    ) # compressed length in bits per pixel

    # failed values
    error = torch.abs(out["x_hat"] - inp)
    failed_value_count = (error > target).sum().item()
    failed_value_ratio = failed_value_count / num_pixels

    # 1. score: >0: failed; <=0: success
    if self.score_type == 'ratio':
      score = error / (target + 1e-6) - 1
    elif self.score_type == 'margin':
      score = error - target
    else:
      raise ValueError(f"Unsupported score_type {self.score_type}")
    
    # 2. surrogate loss for count(score > 0)
    if self.surrogate_loss_type == 'relu':
      is_failed_value = F.relu(score)
    elif self.surrogate_loss_type == 'relu_square':
      is_failed_value = F.relu(score) ** 2
    elif self.surrogate_loss_type == 'softplus':
      is_failed_value = F.softplus(score / self.tau) * self.tau
    elif self.surrogate_loss_type == 'sigmoid':
      is_failed_value = torch.sigmoid(score / self.tau)
    else:
      raise ValueError(f"Unsupported surrogate_loss_type {self.surrogate_loss_type}")

    bpp_failed_value = 32 * is_failed_value.mean() # 32b for float32

    loss = bpp_compression + bpp_failed_value

    return {
      "loss": loss,
      "bpp_compression": bpp_compression.item(),
      "bpp_failed_value": bpp_failed_value.item(),
      "failed_value_ratio": failed_value_ratio,
    }

def normalize_batch_np(data, spread):
  ndim = data.ndim
  batch_size = data.shape[0]
  data_flat = data.reshape(batch_size, -1)
  mins = np.nanmin(data_flat, axis=1)
  maxs = np.nanmax(data_flat, axis=1)
  scales = maxs - mins # [B]
  scales = np.where(scales == 0, 1.0, scales) # if const

  view_shape = (batch_size, ) + (1, ) * (ndim -1)
  data_norm = (data - mins.reshape(view_shape)) / scales.reshape(view_shape)
  spread_norm = spread / scales.reshape(view_shape)

  data_norm = np.where(np.isnan(data_norm), 0.0, data_norm)
  spread_norm = np.where(np.isnan(spread_norm), 1.0, spread_norm)
  spread_norm[spread_norm < 0] = 0

  return data_norm, spread_norm, mins, scales

def denormalize_batch_np(data_norm, mins, scales):
  ndim = data_norm.ndim
  batch_size = data_norm.shape[0]
  view_shape = (batch_size, ) + (1, ) * (ndim -1)
  data = data_norm * scales.reshape(view_shape) + mins.reshape(view_shape)
  return data

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

def configure_optimizers(net, learning_rate, aux_learning_rate):
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
    lr=learning_rate,
  )
  aux_optimizer = optim.Adam(
    (params_dict[n] for n in sorted(aux_parameters)),
    lr=aux_learning_rate,
  )
  return optimizer, aux_optimizer

def train_one_epoch(
  model, 
  criterion, 
  train_dataset, 
  batch_per_epoch,
  optimizer, 
  aux_optimizer, 
  epoch, 
  clip_max_norm, 
  no_aux_loss,
):
  model.train()
  device = next(model.parameters()).device

  batch_idx = 0
  while True:
    data_array, spread_array, is_epoch_finished = train_dataset.sample_batch()

    data_norm, spread_norm, mins, scales = normalize_batch_np(data_array, spread_array)

    data_tensor = torch.from_numpy(data_norm).float().to(device)
    spread_tensor = torch.from_numpy(spread_norm).float().to(device)

    data_tensor = data_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    spread_tensor = spread_tensor.unsqueeze(1).repeat(1, 3, 1, 1)

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    num_params = len(inspect.signature(model.forward).parameters)
    if num_params == 1:
      out_net = model(data_tensor)
    elif num_params == 2:
      out_net = model(data_tensor, spread_tensor)
    else:
      raise ValueError(f"Unsupported number of model.forward parameters: {num_params}")
    
    out_net["x_hat"] = out_net["x_hat"].mean(dim=1)
    out_criterion = criterion(
      out=out_net, 
      inp=data_tensor[:, 0, :, :], 
      target=spread_tensor[:, 0, :, :],
    )
    out_criterion["loss"].backward()

    if clip_max_norm > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    optimizer.step()

    aux_loss = model.aux_loss()
    if not no_aux_loss:
      aux_loss.backward()
      aux_optimizer.step()

    print(
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
      f" Train epoch {epoch} batch {batch_idx}: {data_array.shape}"
      f' Loss: {out_criterion["loss"].item()} |'
      f' Bpp Compression: {out_criterion["bpp_compression"]} |'
      f' Bpp Failed Value: {out_criterion["bpp_failed_value"]} |'
      f' Failed Value Ratio: {out_criterion["failed_value_ratio"]} |'
      f" Aux loss: {aux_loss.item()}"
    )

    # end of epoch
    batch_idx += 1
    if batch_per_epoch < 0:
      if is_epoch_finished:
        break
    else:
      if batch_idx >= batch_per_epoch:
        break

def valid_epoch(
  model, 
  criterion,
  valid_dataset,
  epoch,
  interval,
):
  model.eval()
  device = next(model.parameters()).device

  avg_metrics = {}

  with torch.no_grad():
    while True:
      data_array, spread_array, is_epoch_finished = valid_dataset.sample_batch()
      data_array = data_array[::interval]
      spread_array = spread_array[::interval]

      data_norm, spread_norm, mins, scales = normalize_batch_np(data_array, spread_array)

      data_tensor = torch.from_numpy(data_norm).float().to(device)
      spread_tensor = torch.from_numpy(spread_norm).float().to(device)

      data_tensor = data_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
      spread_tensor = spread_tensor.unsqueeze(1).repeat(1, 3, 1, 1)

      num_params = len(inspect.signature(model.forward).parameters)
      if num_params == 1:
        out_net = model(data_tensor)
        out_enc = model.compress(data_tensor)
      elif num_params == 2:
        out_net = model(data_tensor, spread_tensor)
        out_enc = model.compress(data_tensor, spread_tensor)
      else:
        raise ValueError(f"Unsupported number of model.forward parameters: {num_params}")

      # inference
      out_net["x_hat"] = out_net["x_hat"].mean(dim=1)
      out_criterion = criterion(
        out=out_net, 
        inp=data_tensor[:, 0, :, :], 
        target=spread_tensor[:, 0, :, :],
      )

      avg_metrics.setdefault("loss", AverageMeter()).update(out_criterion["loss"].item())
      avg_metrics.setdefault("bpp_compression", AverageMeter()).update(out_criterion["bpp_compression"])
      avg_metrics.setdefault("bpp_failed_value", AverageMeter()).update(out_criterion["bpp_failed_value"])
      avg_metrics.setdefault("failed_value_ratio", AverageMeter()).update(out_criterion["failed_value_ratio"])
      avg_metrics.setdefault("aux_loss", AverageMeter()).update(model.aux_loss().item())

      # real compression
      out_enc["shape"] = list(out_enc["shape"])
      out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
      out_dec["x_hat"] = out_dec["x_hat"].mean(dim=1)

      data_hat = denormalize_batch_np(out_dec["x_hat"].detach().cpu().numpy(), mins, scales)
      error = np.abs(data_hat - data_array)
      fail_mask = error > spread_array
      fail_idx = np.flatnonzero(fail_mask).astype(np.int32)
      fail_val = data_array.flat[fail_idx]
      packed_fail_mask = np.packbits(fail_mask.ravel())
      compressed_fail_mask = zlib.compress(packed_fail_mask.tobytes(), level=6)
      compressed_fail_idx = zlib.compress(fail_idx.tobytes(), level=6)
      compressed_fail_val = zlib.compress(fail_val.tobytes(), level=6)

      if len(compressed_fail_mask) <= len(compressed_fail_idx):
        failed_value_compressed_size_bytes = len(compressed_fail_mask) + len(compressed_fail_val)
        compressed_fail_info = {
          "fail_mask": compressed_fail_mask,
          "fail_val": compressed_fail_val,
        }
      else:
        failed_value_compressed_size_bytes = len(compressed_fail_idx) + len(compressed_fail_val)
        compressed_fail_info = {
          "fail_idx": compressed_fail_idx,
          "fail_val": compressed_fail_val,
        }

      failed_value_compression_ratio = fail_val.nbytes / failed_value_compressed_size_bytes
      compressed_bitstream = pickle.dumps([compressed_fail_info, out_enc])
      data_size_bytes = data_array.nbytes
      compressed_size_bytes = len(compressed_bitstream)
      real_bpp = (compressed_size_bytes * 8) / np.prod(data_array.shape)
      real_compression_ratio = data_size_bytes / compressed_size_bytes

      avg_metrics.setdefault("real_compression_ratio", AverageMeter()).update(real_compression_ratio)
      avg_metrics.setdefault("real_bpp", AverageMeter()).update(real_bpp)
      avg_metrics.setdefault("real_failed_value_compression_ratio", AverageMeter()).update(failed_value_compression_ratio)

      if is_epoch_finished:
        break

  result = {
    "epoch": epoch,
    k: v.avg for k, v in avg_metrics.items(),
  }

  print(
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    f" Valid epoch {epoch} average metrics:"
    f" Loss: {result['loss']} |"
    f' Bpp Compression: {result["bpp_compression"]} |'
    f' Bpp Failed Value: {result["bpp_failed_value"]} |'
    f' Failed Value Ratio: {result["failed_value_ratio"]} |'
    f" Aux loss: {avg_metrics['aux_loss']}"
    f' Real Compression Ratio: {result["real_compression_ratio"]} |'
    f' Real Bpp: {result["real_bpp"]} |'
    f' Real Failed Value Compression Ratio: {result["real_failed_value_compression_ratio"]}'
  )

  return result


def save_checkpoint(state, is_best, epoch, save_path, filename):
  torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
  if epoch % 5 == 0:
    torch.save(state, os.path.join(save_path, filename))
  if is_best:
    torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))


def main(
  # model
  model,
  N,
  use_bound_in_decoder,
  # dataset
  variable,
  # training
  epochs,
  batch_size,
  batch_per_epoch,
  learning_rate,
  clip_max_norm,
  lr_epoch,
  # loss
  score_type,
  surrogate_loss_type,
  tau,
  # ckpt
  save_path,
  checkpoint=None,
  continue_train=False,
  freeze_pretrained_modules=False,
  seed=42,
):
  aux_learning_rate = 1e-3
  if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

  save_path = os.path.join(save_path, f"N_{N}_model_{model}_var_{variable}_score_{score_type}_surrogate_{surrogate_loss_type}_tau_{tau}_lr_{learning_rate}_bs_{batch_size}_seed_{seed}")
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # datasets
  patch_size = -1
  padding_factor = 128
  n_files_per_load = 4
  loader_mode = 'thread'
  train_dataset = Era5ReanalysisDatasetSingleLevel(
    variable=variable,
    batch_size=batch_size,
    patch_size=patch_size,
    padding_factor=padding_factor,
    split="train",
    n_files_per_load=n_files_per_load,
    loader_mode=loader_mode,
  )

  valid_dataset = Era5ReanalysisDatasetSingleLevel(
    variable=variable,
    batch_size=60,
    patch_size=-1,
    padding_factor=padding_factor,
    split="valid",
    n_files_per_load=n_files_per_load,
    loader_mode=loader_mode,
  )

  device = 'cuda:0'

  if model == 'tcm':
    print("[INFO] Using TCM model")
    net = TCM(
      config=[2,2,2,2,2,2], 
      head_dim=[8, 16, 32, 32, 16, 8], 
      drop_path_rate=0.0, 
      N=N, 
      M=320
    )
  elif model == 'tcm_weighted':
    print("[INFO] Using TCMWeighted model")
    net = TCMWeighted(
      config=[2,2,2,2,2,2], 
      head_dim=[8, 16, 32, 32, 16, 8], 
      drop_path_rate=0.0, 
      N=N, 
      M=320, 
      use_bound_in_decoder=use_bound_in_decoder
    )
  else:
    raise ValueError(f"Unsupported model {model}")

  # freeze pretrained modules
  if freeze_pretrained_modules:
    assert model == 'TCMWeighted', "Only TCMWeighted model supports freezing pretrained modules"
    print("[INFO] Pretrained modules are freezed")
    net.freeze_pretrained_modules = True
  net = net.to(device)

  optimizer, aux_optimizer = configure_optimizers(net, learning_rate, aux_learning_rate)
  milestones = lr_epoch
  print("milestones: ", milestones)
  lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

  criterion = ErrorBoundedRateOnlyLoss(score_type, surrogate_loss_type, tau)
  criterion = criterion.to(device)

  last_epoch = 0
  if checkpoint:
    checkpoint = torch.load(checkpoint, map_location=device)
    # handle DP
    dictory = {}
    for k, v in checkpoint["state_dict"].items():
      dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)

    if continue_train:
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
        f"-1_checkpoint.pth.tar",
      )
    print(f"Loaded {checkpoint} and continue train from epoch {last_epoch}, ({continue_train=})")
  else:
    print("Training from scratch")

  if torch.cuda.device_count() > 1:
    net = CustomDataParallel(net)
    print(f'[INFO] DataParallel: Using {torch.cuda.device_count()} GPUs') 

  if model == 'tcm_weighted':
    no_aux_loss = freeze_pretrained_modules and not use_bound_in_decoder
    print(f"[INFO] no_aux_loss: {no_aux_loss}")
  else:
    no_aux_loss = False

  best_loss = float("inf")
  results = []
  for epoch in range(last_epoch, epochs):
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    train_one_epoch(
      model=net, 
      criterion=criterion, 
      train_dataset=train_dataset, 
      batch_per_epoch=batch_per_epoch,
      optimizer=optimizer, 
      aux_optimizer=aux_optimizer, 
      epoch=epoch, 
      clip_max_norm=clip_max_norm, 
      no_aux_loss=no_aux_loss,
    )
    result = valid_epoch(
      model=net, 
      criterion=criterion,
      valid_dataset=valid_dataset,
      epoch=epoch,
      interval=6,
    )

    results.append(result)
    loss = result["loss"]
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
      f"{epoch}_checkpoint.pth.tar",
    )


if __name__ == "__main__":
  fire.Fire(main)