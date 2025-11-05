SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

N=128
lambda=0.05
lr=1e-4

use_gray_scale_weight=0
use_log_norm_weight=0
use_weight_in_decoder=1

python -u train_weighted.py \
    --dataset /capstor/store/cscs/userlab/g34/ljiayong/datasets/ILSVRC-kaggle/100k \
    --epochs 10 \
    --lr_epoch 11 \
    --learning-rate $lr \
    --N $N \
    --lambda $lambda \
    --batch-size 64 \
    --valid-batch-size 128 \
    --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained_tcm_weighted_wd${use_weight_in_decoder}/converted_tcm_weighted_lic_tcm_n_${N}_lambda_${lambda}.pth.tar \
    --save-path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_100k_${lr}_gw${use_gray_scale_weight}_lw${use_log_norm_weight}_wd${use_weight_in_decoder} \
    --use-gray-scale-weight $use_gray_scale_weight \
    --use-log-norm-weight $use_log_norm_weight \
    --use-weight-in-decoder $use_weight_in_decoder \
    --freeze-pretrained
