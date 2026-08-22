#!/bin/bash
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=compare_all_%j.log
#SBATCH --job-name=compare-all

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

python ~/thesis-xai/7_compare_all.py

echo "Job finished: $(date)"
