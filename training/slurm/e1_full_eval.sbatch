#!/bin/bash
#SBATCH --job-name=scout-E1-full-eval
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/E1-full-eval-%j.out

set -e
cd ~/scout

module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

python training/eval.py \
    --model results/E1_full/weights/best.pt \
    --data ~/scratch/scout_data \
    --split val \
    --n 500 \
    --name E1_full_eval \
    --batch 16
