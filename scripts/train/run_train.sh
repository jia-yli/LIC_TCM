SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

lambdas=(0.0025 0.0067 0.025 0.05)

for i in "${!lambdas[@]}"; do
  lambda=${lambdas[$i]}
  gpu_id=$i

  echo "Launching training on GPU $gpu_id with lambda $lambda"

  CUDA_VISIBLE_DEVICES=$gpu_id python -u ${SCRIPT_DIR}/../../train.py -d /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/datasets/imagenet_256_300000 \
    --cuda --N 128 --lambda $lambda --epochs 50 --lr_epoch 45 48 \
    --save_path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints --save &

done

wait
