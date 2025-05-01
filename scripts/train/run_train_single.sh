SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CUDA_VISIBLE_DEVICES='0' python -u ${SCRIPT_DIR}/../../train.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_300000 \
  --cuda --N 128 --lambda 0.05 --epochs 50 --lr_epoch 45 48 \
  --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints --save 
