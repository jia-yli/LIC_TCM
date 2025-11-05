#!/usr/bin/env bash
set -euo pipefail

SBATCH_FILE="/users/ljiayong/projects/LIC_TCM/scripts/train/launch_train_weighted.sbatch"

# Comment out lines to skip any combination.

# gw=0, lw=0, wd=0
sbatch --job-name="lic-tcm-train-gw0-lw0-wd0" \
  --export=ALL,use_gray_scale_weight=0,use_log_norm_weight=0,use_weight_in_decoder=0 \
  "${SBATCH_FILE}"

# gw=0, lw=0, wd=1
sbatch --job-name="lic-tcm-train-gw0-lw0-wd1" \
  --export=ALL,use_gray_scale_weight=0,use_log_norm_weight=0,use_weight_in_decoder=1 \
  "${SBATCH_FILE}"

# gw=0, lw=1, wd=0
sbatch --job-name="lic-tcm-train-gw0-lw1-wd0" \
  --export=ALL,use_gray_scale_weight=0,use_log_norm_weight=1,use_weight_in_decoder=0 \
  "${SBATCH_FILE}"

# gw=0, lw=1, wd=1
sbatch --job-name="lic-tcm-train-gw0-lw1-wd1" \
  --export=ALL,use_gray_scale_weight=0,use_log_norm_weight=1,use_weight_in_decoder=1 \
  "${SBATCH_FILE}"

# gw=1, lw=0, wd=0
sbatch --job-name="lic-tcm-train-gw1-lw0-wd0" \
  --export=ALL,use_gray_scale_weight=1,use_log_norm_weight=0,use_weight_in_decoder=0 \
  "${SBATCH_FILE}"

# gw=1, lw=0, wd=1
sbatch --job-name="lic-tcm-train-gw1-lw0-wd1" \
  --export=ALL,use_gray_scale_weight=1,use_log_norm_weight=0,use_weight_in_decoder=1 \
  "${SBATCH_FILE}"

# gw=1, lw=1, wd=0
sbatch --job-name="lic-tcm-train-gw1-lw1-wd0" \
  --export=ALL,use_gray_scale_weight=1,use_log_norm_weight=1,use_weight_in_decoder=0 \
  "${SBATCH_FILE}"

# gw=1, lw=1, wd=1
sbatch --job-name="lic-tcm-train-gw1-lw1-wd1" \
  --export=ALL,use_gray_scale_weight=1,use_log_norm_weight=1,use_weight_in_decoder=1 \
  "${SBATCH_FILE}"