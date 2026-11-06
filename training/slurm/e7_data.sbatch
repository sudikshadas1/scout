#!/bin/bash
# E7 data: download the selected Wellington pictures and build the training list. Safe to re-run; skips what it has.
#   sbatch training/slurm/e7_data.sbatch
#SBATCH --job-name=scout-e7-data
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

DATA=~/scratch/scout_data
python data/add_wellington.py --data "$DATA"
python data/make_train_list.py --data "$DATA" --cat-repeat 1 --extra-dir "$DATA/wct/yolo/images/train" --tag e7
