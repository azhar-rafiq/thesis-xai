#!/bin/bash
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00
#SBATCH --mem=490G
#SBATCH --cpus-per-task=8
#SBATCH --output=train_cnn_optuna_%j.log
#SBATCH --job-name=cnn_optuna

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load bluebear
module load bear-apps/2024a/live
module load CUDA/12.6.0
module load cuDNN/9.8.0.87-CUDA-12.6.0

source ~/thesis-xai/thesis-venv/bin/activate

python ~/thesis-xai/1b_train_cnn_optuna.py

echo "Job finished: $(date)"
