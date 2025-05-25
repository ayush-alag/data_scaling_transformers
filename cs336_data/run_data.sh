#!/bin/bash
#SBATCH --job-name=cc-filter
#SBATCH --output=logs/filter_%j.out
#SBATCH --error=logs/filter_%j.out
#SBATCH --time=12:00:00        # maximum walltime
#SBATCH --account=student
#SBATCH --partition=a4-batch
#SBATCH --qos=a4-batch-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

cd ~/data_scaling_transformers

echo "Starting filter driver at $(date)"
uv run python cs336_data/filter_cc_files.py
uv run python cs336_data/tokenize_filtered.py

echo "Driver exited at $(date)"