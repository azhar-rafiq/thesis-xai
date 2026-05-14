#!/bin/bash
#SBATCH --job-name=dicom_index
#SBATCH --account=karwatha-karwath-hds-pg-research
#SBATCH --qos=bbdefault
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=dicom_index_%j.out

module purge
module load bear-apps/2023a/live
module load Python/3.11.3-GCCcore-12.3.0

python -m pip install pydicom tqdm pandas --user --quiet

python build_dicom_index.py