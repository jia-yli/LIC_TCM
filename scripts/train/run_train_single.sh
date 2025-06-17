SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CUDA_VISIBLE_DEVICES='0' python -u ${SCRIPT_DIR}/../../train_era5.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_300000 \
  --patch-size 256 --batch-size 8 \
  --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_0.05.pth.tar \
  --cuda --N 64 --lambda 0.05 --epochs 10 --lr_epoch 11 \
  --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints --save 
