SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

N=64
lambda=0.05
VAR="$1"

python -u ${SCRIPT_DIR}/../../train_era5_weighted.py --variable "${VAR}" \
  --patch-size -1 --batch-size 4 \
  --continue-train \
  --cuda --N ${N} --lambda ${lambda} --epochs 20 --lr_epoch 21 \
  --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_rand_init --save