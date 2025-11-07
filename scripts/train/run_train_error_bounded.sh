SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pip install fire

# tcm
model=tcm
use_bound_in_decoder=0

variable=2m_temperature
batch_size=8
batch_per_epoch=100
lr=1e-4
clip_max_norm=1.0

score_type=ratio
surrogate_loss_type=sigmoid
tau=0.05

save_path=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_error_bounded_bs_${batch_size}_be_${batch_per_epoch}_lr_${lr}_cn_${clip_max_norm}
checkpoint=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar

python -u train_era5_error_bounded.py \
  --model ${model} \
  --use-bound-in-decoder ${use_bound_in_decoder} \
  --variable ${variable} \
  --epochs 10 \
  --batch-size ${batch_size} \
  --batch-per-epoch ${batch_per_epoch} \
  --learning-rate ${lr} \
  --clip-max-norm ${clip_max_norm} \
  --lr-epoch 11 \
  --score-type ${score_type} \
  --surrogate-loss-type ${surrogate_loss_type} \
  --tau ${tau} \
  --save-path ${save_path} \
  --checkpoint ${checkpoint}

# tcm weighted
# model=tcm_weighted
# use_bound_in_decoder=0

# variable=2m_temperature
# batch_size=8
# batch_per_epoch=100
# lr=1e-4
# clip_max_norm=1.0

# score_type=ratio
# surrogate_loss_type=relu
# tau=0.05

# save_path=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_error_bounded_bs_${batch_size}_be_${batch_per_epoch}_lr_${lr}_cn_${clip_max_norm}
# checkpoint=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_128_lambda_0.05.pth.tar

# python -u train_era5_error_bounded.py \
#   --model ${model} \
#   --use-bound-in-decoder ${use_bound_in_decoder} \
#   --variable ${variable} \
#   --epochs 10 \
#   --batch-size ${batch_size} \
#   --batch-per-epoch ${batch_per_epoch} \
#   --learning-rate ${lr} \
#   --clip-max-norm ${clip_max_norm} \
#   --lr-epoch 11 \
#   --score-type ${score_type} \
#   --surrogate-loss-type ${surrogate_loss_type} \
#   --tau ${tau} \
#   --save-path ${save_path} \
#   --checkpoint ${checkpoint}