#!/bin/bash
#SBATCH --account=a-g34                   # account to use
#SBATCH --job-name=lic-tcm-train-eb          # Job name
#SBATCH --output=./slurm_out/job_%j_out.txt   # Output file (job ID and task number)
#SBATCH --error=./slurm_out/job_%j_err.txt    # Error file (job ID and task number)
#SBATCH --partition=normal                # Partitions to submit the job
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=128               # Number of CPU cores per task
#SBATCH --mem=384G                        # Memory allocation
#SBATCH --gpus=4                          # Number of GPUs
#SBATCH --time=12:00:00                    # Time limit

# Print detailed job metadata
JOB_TIME_LIMIT=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)
CURRENT_TIMESTAMP_FULL=$(date --utc)
CURRENT_TIMESTAMP_SHORT=$(date --utc -d "$CURRENT_TIMESTAMP_FULL" +"%Y%m%d_%H%M%S")
echo "================== JOB METADATA =================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "User ID: $SLURM_JOB_USER"
echo "Job Submit Directory: $SLURM_SUBMIT_DIR"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory Allocated: $SLURM_MEM_PER_NODE"
echo "GPUs Allocated: $SLURM_GPUS"
echo "Time Limit: $JOB_TIME_LIMIT"
echo "Job Start Time: $CURRENT_TIMESTAMP_FULL"
echo "==================================================="

ENV_VARS="
  export TQDM_MININTERVAL=120
  # export TQDM_DISABLE=1
"

model="${model:-tcm}"
N="${N:-64}"
use_bound_in_decoder="${use_bound_in_decoder:-0}"
variable="${variable:-2m_temperature}"
batch_size="${batch_size:-8}"
batch_per_epoch="${batch_per_epoch:-100}"
lr="${lr:-1e-4}"
clip_max_norm="${clip_max_norm:-1.0}"
score_type="${score_type:-ratio}"
surrogate_loss_type="${surrogate_loss_type:-sigmoid}"
tau="${tau:-0.05}"
seed="${seed:-42}"

echo "----- TRAIN PARAMS RECEIVED -----"
echo "model                   = ${model}"
echo "N                       = ${N}"
echo "use_bound_in_decoder    = ${use_bound_in_decoder}"
echo "variable                = ${variable}"
echo "batch_size              = ${batch_size}"
echo "batch_per_epoch         = ${batch_per_epoch}"
echo "learning-rate           = ${lr}"
echo "clip_max_norm           = ${clip_max_norm}"
echo "score_type              = ${score_type}"
echo "surrogate_loss_type     = ${surrogate_loss_type}"
echo "tau                     = ${tau}"
echo "seed                    = ${seed}"
echo "---------------------------------"

if [ "${model}" = "tcm" ]; then
  # TCM model - single stage training
  save_path=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_${model}_error_bounded_bs_${batch_size}_be_${batch_per_epoch}_lr_${lr}_cn_${clip_max_norm}
  checkpoint=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained/lic_tcm_n_${N}_lambda_0.05.pth.tar

  srun --environment=/users/ljiayong/projects/LIC_TCM/docker/alps/clariden.toml bash -c "
    ${ENV_VARS}
    date --utc
    python -u train_era5_error_bounded.py \
      --model ${model} \
      --N ${N} \
      --use-bound-in-decoder ${use_bound_in_decoder} \
      --variable ${variable} \
      --epochs 80 \
      --batch-size ${batch_size} \
      --batch-per-epoch ${batch_per_epoch} \
      --learning-rate ${lr} \
      --clip-max-norm ${clip_max_norm} \
      --lr-epoch 81 \
      --score-type ${score_type} \
      --surrogate-loss-type ${surrogate_loss_type} \
      --tau ${tau} \
      --save-path ${save_path} \
      --checkpoint ${checkpoint} \
      --seed ${seed}
    date --utc
  "

elif [ "${model}" = "tcm_weighted" ]; then
  # TCM Weighted model - stage 1 (freeze pretrained) + stage 2 (full training)
  
  # Stage 1: freeze pretrained modules
  save_path=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_${model}_error_bounded_stage1_bs_${batch_size}_be_${batch_per_epoch}_lr_${lr}_cn_${clip_max_norm}
  checkpoint=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/pretrained_tcm_weighted_wd${use_bound_in_decoder}/converted_tcm_weighted_lic_tcm_n_${N}_lambda_0.05.pth.tar

  srun --environment=/users/ljiayong/projects/LIC_TCM/docker/alps/clariden.toml bash -c "
    ${ENV_VARS}
    date --utc
    python -u train_era5_error_bounded.py \
      --model ${model} \
      --N ${N} \
      --use-bound-in-decoder ${use_bound_in_decoder} \
      --variable ${variable} \
      --epochs 20 \
      --batch-size ${batch_size} \
      --batch-per-epoch ${batch_per_epoch} \
      --learning-rate ${lr} \
      --clip-max-norm ${clip_max_norm} \
      --lr-epoch 21 \
      --score-type ${score_type} \
      --surrogate-loss-type ${surrogate_loss_type} \
      --tau ${tau} \
      --save-path ${save_path} \
      --checkpoint ${checkpoint} \
      --freeze-pretrained-modules \
      --seed ${seed}
    date --utc
  "

  # Stage 2: full training from stage 1 checkpoint
  save_path=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_${model}_error_bounded_stage2_bs_${batch_size}_be_${batch_per_epoch}_lr_${lr}_cn_${clip_max_norm}
  checkpoint=/capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_${model}_error_bounded_stage1_bs_${batch_size}_be_${batch_per_epoch}_lr_${lr}_cn_${clip_max_norm}
  checkpoint=${checkpoint}/N_${N}_bd_${use_bound_in_decoder}_var_${variable}_score_${score_type}_surrogate_${surrogate_loss_type}_tau_${tau}_seed_${seed}/checkpoint_best.pth.tar

  srun --environment=/users/ljiayong/projects/LIC_TCM/docker/alps/clariden.toml bash -c "
    ${ENV_VARS}
    date --utc
    python -u train_era5_error_bounded.py \
      --model ${model} \
      --N ${N} \
      --use-bound-in-decoder ${use_bound_in_decoder} \
      --variable ${variable} \
      --epochs 40 \
      --batch-size ${batch_size} \
      --batch-per-epoch ${batch_per_epoch} \
      --learning-rate ${lr} \
      --clip-max-norm ${clip_max_norm} \
      --lr-epoch 41 \
      --score-type ${score_type} \
      --surrogate-loss-type ${surrogate_loss_type} \
      --tau ${tau} \
      --save-path ${save_path} \
      --checkpoint ${checkpoint} \
      --seed ${seed}
    date --utc
  "

fi
