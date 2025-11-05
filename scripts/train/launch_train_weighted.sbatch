#!/bin/bash
#SBATCH --account=a-g34                   # account to use
#SBATCH --job-name=lic-tcm-train             # Job name
#SBATCH --output=./slurm_out/job_%j_out.txt   # Output file (job ID and task number)
#SBATCH --error=./slurm_out/job_%j_err.txt    # Error file (job ID and task number)
#SBATCH --partition=normal                # Partitions to submit the job
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=128               # Number of CPU cores per task
#SBATCH --mem=256G                        # Memory allocation
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
N=128
lambda=0.05
lr=1e-4
use_gray_scale_weight="${use_gray_scale_weight:-1}"
use_log_norm_weight="${use_log_norm_weight:-1}"
use_weight_in_decoder="${use_weight_in_decoder:-1}"

echo "----- TRAIN PARAMS RECEIVED -----"
echo "N                       = ${N}"
echo "lambda                  = ${lambda}"
echo "learning-rate           = ${lr}"
echo "use_gray_scale_weight   = ${use_gray_scale_weight}"
echo "use_log_norm_weight     = ${use_log_norm_weight}"
echo "use_weight_in_decoder   = ${use_weight_in_decoder}"
echo "---------------------------------"

srun --environment=/users/ljiayong/projects/LIC_TCM/docker/alps/clariden.toml bash -c "
  ${ENV_VARS}
  date --utc
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
  date --utc
"

# srun --environment=/users/ljiayong/projects/LIC_TCM/docker/alps/clariden.toml bash -c "
#   ${ENV_VARS}
#   date --utc
#   python -u train_weighted.py \
#     --dataset /capstor/store/cscs/userlab/g34/ljiayong/datasets/ILSVRC-kaggle/100k \
#     --epochs 10 \
#     --lr_epoch 11 \
#     --learning-rate $lr \
#     --N $N \
#     --lambda $lambda \
#     --batch-size 64 \
#     --valid-batch-size 128 \
#     --checkpoint /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_100k_${lr}_gw${use_gray_scale_weight}_lw${use_log_norm_weight}_wd${use_weight_in_decoder}/N_${N}_lambda_${lambda}/checkpoint_best.pth.tar \
#     --save-path /capstor/scratch/cscs/ljiayong/workspace/LIC_TCM/checkpoints_tcm_weighted_100k_${lr}_gw${use_gray_scale_weight}_lw${use_log_norm_weight}_wd${use_weight_in_decoder}_stage2_64 \
#   date --utc
# "
