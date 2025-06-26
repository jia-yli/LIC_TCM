SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

N=64
lambda=0.05

# python -u ${SCRIPT_DIR}/../../train_era5.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_50000 \
#   --patch-size -1 --batch-size 2 \
#   --continue-train \
#   --cuda --N 64 --lambda $lambda --epochs 4 --lr_epoch 5 \
#   --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints --save &

python -u ${SCRIPT_DIR}/../../train_era5_weighted.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_50000 \
  --patch-size -1 --batch-size 8 \
  --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_${N}_lambda_${lambda}.pth.tar \
  --cuda --N ${N} --lambda ${lambda} --epochs 4 --lr_epoch 5 \
  --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_weighted_fintune --save

