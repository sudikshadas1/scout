#!/bin/bash
# Train one experiment on PACE ICE with the account/partition that worked for E1 (training/slurm/e1_full.sbatch).
# Purpose: E1b, E1c and later runs go through training/train_yolo.py so settings are recorded in summary.json.
# Status:  Header copied from the working e1_full.sbatch. Not yet run with train_yolo.py.
#   sbatch -J scout-E1b training/slurm/train_pace.sbatch E1b --epochs 80 --patience 25
#   sbatch -J scout-E1c training/slurm/train_pace.sbatch E1c --epochs 80 --patience 25 \
#          --data-yaml ~/scratch/scout_data/scout_cat2x.yaml
#SBATCH --job-name=scout-train
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
NAME=${1:?usage: sbatch train_pace.sbatch <experiment-name> [train_yolo.py args]}
shift
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

# E1 used data/scout_pace.yaml (absolute scratch path); default to it unless --data-yaml is passed.
python training/train_yolo.py --name "$NAME" --data-yaml data/scout_pace.yaml \
    --batch 16 --workers 4 --device 0 "$@"
