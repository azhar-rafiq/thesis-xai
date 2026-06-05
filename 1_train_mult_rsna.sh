#!/bin/bash
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=490G
#SBATCH --cpus-per-task=8
#SBATCH --output=train_mult_rsna_%j.log
#SBATCH --job-name=rsna-train

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load bluebear
module load bear-apps/2024a/live
module load CUDA/12.6.0
module load cuDNN/9.8.0.87-CUDA-12.6.0

source ~/thesis-xai/thesis-venv/bin/activate

python ~/thesis-xai/1_train_mult_rsna.py

echo "Job finished: $(date)"
