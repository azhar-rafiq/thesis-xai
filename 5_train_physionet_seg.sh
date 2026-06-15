#!/bin/bash
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=physionet_seg_%j.log
#SBATCH --job-name=physionet-unet

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load bluebear
module load bear-apps/2024a/live
module load CUDA/12.6.0
module load cuDNN/9.8.0.87-CUDA-12.6.0

source ~/thesis-xai/thesis-venv/bin/activate

python -c "import nibabel" 2>/dev/null || pip install nibabel --quiet

python ~/thesis-xai/5_train_physionet_seg.py

echo "Job finished: $(date)"
