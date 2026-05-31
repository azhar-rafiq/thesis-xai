#!/bin/bash
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbdefault
#SBATCH --time=12:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --output=preprocess_%j.log
#SBATCH --job-name=rsna-preprocess

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load bluebear

source ~/thesis-xai/thesis-venv/bin/activate

python ~/thesis-xai/0_preprocess_rsna.py

echo "Job finished: $(date)"