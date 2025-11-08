#!/usr/bin/env bash
set -euo pipefail

SBATCH_FILE="/users/ljiayong/projects/LIC_TCM/scripts/train/launch_train_error_bounded.sbatch"

# Comment out lines to skip any combination.
# TCM model (batch_size=8)
model=tcm
N_list=(64 128)
batch_size=8
batch_per_epoch=1000

score_types=("ratio" "margin")
surrogate_loss_types_no_tau=("relu")
surrogate_loss_types_with_tau=("softplus" "sigmoid")
taus=(0.05 0.1)
# no tau
for N in "${N_list[@]}"; do
  for score_type in "${score_types[@]}"; do
    for surrogate_loss_type in "${surrogate_loss_types_no_tau[@]}"; do
      tau=0.05
      sbatch --job-name="lic-tcm-train-eb-tcm" \
        --export=ALL,model=${model},N=${N},batch_size=${batch_size},batch_per_epoch=${batch_per_epoch},score_type=${score_type},surrogate_loss_type=${surrogate_loss_type},tau=${tau} \
      "${SBATCH_FILE}"
    done
  done
done
# with tau
for N in "${N_list[@]}"; do
  for score_type in "${score_types[@]}"; do
    for surrogate_loss_type in "${surrogate_loss_types_with_tau[@]}"; do
      for tau in "${taus[@]}"; do
        sbatch --job-name="lic-tcm-train-eb-tcm" \
          --export=ALL,model=${model},N=${N},batch_size=${batch_size},batch_per_epoch=${batch_per_epoch},score_type=${score_type},surrogate_loss_type=${surrogate_loss_type},tau=${tau} \
        "${SBATCH_FILE}"
      done
    done
  done
done

# TCM Weighted model with use_bound_in_decoder=0 (batch_size=4)
# model=tcm_weighted
# batch_size=4
# batch_per_epoch=500

# use_bound_in_decoder_list=(0 1)
# score_types=("ratio" "margin")
# surrogate_loss_types_no_tau=("relu" "relu_square")
# surrogate_loss_types_with_tau=("softplus" "sigmoid")
# taus=(0.05 0.1)

# # no tau
# for score_type in "${score_types[@]}"; do
#   for surrogate_loss_type in "${surrogate_loss_types_no_tau[@]}"; do
#     for use_bound_in_decoder in "${use_bound_in_decoder_list[@]}"; do
#       tau=0.05
#       sbatch --job-name="lic-tcm-train-eb-weighted-bd${use_bound_in_decoder}" \
#         --export=ALL,model=${model},use_bound_in_decoder=${use_bound_in_decoder},batch_size=${batch_size},batch_per_epoch=${batch_per_epoch},score_type=${score_type},surrogate_loss_type=${surrogate_loss_type},tau=${tau} \
#         "${SBATCH_FILE}"
#     done
#   done
# done
# # with tau
# for score_type in "${score_types[@]}"; do
#   for surrogate_loss_type in "${surrogate_loss_types_with_tau[@]}"; do
#     for tau in "${taus[@]}"; do
#       for use_bound_in_decoder in "${use_bound_in_decoder_list[@]}"; do
#         sbatch --job-name="lic-tcm-train-eb-weighted-bd${use_bound_in_decoder}" \
#           --export=ALL,model=${model},use_bound_in_decoder=${use_bound_in_decoder},batch_size=${batch_size},batch_per_epoch=${batch_per_epoch},score_type=${score_type},surrogate_loss_type=${surrogate_loss_type},tau=${tau} \
#           "${SBATCH_FILE}"
#       done
#     done
#   done
# done
