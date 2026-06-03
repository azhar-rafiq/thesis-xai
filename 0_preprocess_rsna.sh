#!/bin/bash
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbdefault
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=preprocess_%j.log
#SBATCH --job-name=rsna-preprocess

set -e

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load bluebear

source ~/thesis-xai/thesis-venv/bin/activate

# parallelism is handled by multiprocessing workers, not threads
# setting OMP_NUM_THREADS=1 prevents each worker from spawning extra threads
export OMP_NUM_THREADS=1

python -c "from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit" \
    || { echo "ERROR: iterstrat not installed - run: pip install iterative-stratification"; exit 1; }

python ~/thesis-xai/0_preprocess_rsna.py

echo "Job finished: $(date)"
