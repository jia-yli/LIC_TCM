SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

lambdas=(0.0025 0.0067 0.025 0.05)

for i in "${!lambdas[@]}"; do
  lambda=${lambdas[$i]}
  gpu_id=$i

  echo "Launching training on GPU $gpu_id with lambda $lambda"

  CUDA_VISIBLE_DEVICES=$gpu_id python -u ${SCRIPT_DIR}/../../train_era5.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_50000 \
    --patch-size -1 --batch-size 2 \
    --continue-train \
    --cuda --N 64 --lambda $lambda --epochs 4 --lr_epoch 5 \
    --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints --save &

  # CUDA_VISIBLE_DEVICES=$gpu_id python -u ${SCRIPT_DIR}/../../train_era5.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_50000 \
  #   --patch-size -1 --batch-size 2 \
  #   --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_64_lambda_${lambda}.pth.tar \
  #   --cuda --N 64 --lambda ${lambda} --epochs 4 --lr_epoch 5 \
  #   --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints --save &

done

wait
