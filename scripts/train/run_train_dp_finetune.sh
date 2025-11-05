SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

N=64
lambda=0.05
VAR="$1"

# python -u ${SCRIPT_DIR}/../../train_era5_weighted.py --variable "${VAR}" \
#   --patch-size -1 --batch-size 4 \
#   --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained_tcm_weighted/converted_tcm_weighted_lic_tcm_n_${N}_lambda_${lambda}.pth.tar \
#   --freeze-tcm \
#   --cuda --N ${N} --lambda ${lambda} --epochs 10 --lr_epoch 11 \
#   --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_finetune --save

# python -u ${SCRIPT_DIR}/../../train_era5_weighted.py --variable "${VAR}" \
#   --patch-size -1 --batch-size 4 \
#   --continue-train \
#   --freeze-tcm \
#   --cuda --N ${N} --lambda ${lambda} --epochs 10 --lr_epoch 11 \
#   --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_finetune --save

# python -u ${SCRIPT_DIR}/../../train_era5_weighted.py --variable "${VAR}" \
#   --patch-size -1 --batch-size 4 \
#   --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_finetune_freeze/N_${N}_lambda_${lambda}_${VAR}/checkpoint_best.pth.tar \
#   --cuda --N ${N} --lambda ${lambda} --epochs 10 --lr_epoch 11 \
#   --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_finetune --save

python -u ${SCRIPT_DIR}/../../train_era5_weighted.py --variable "${VAR}" \
  --patch-size -1 --batch-size 4 \
  --continue-train \
  --cuda --N ${N} --lambda ${lambda} --epochs 10 --lr_epoch 11 \
  --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_finetune --save
