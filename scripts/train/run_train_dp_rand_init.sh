SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

N=64
lambda=0.05

python -u ${SCRIPT_DIR}/../../train_era5_weighted.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_50000 \
  --patch-size -1 --batch-size 8 \
  --continue-train \
  --cuda --N ${N} --lambda ${lambda} --epochs 10 --lr_epoch 11 \
  --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_rand_init --save

