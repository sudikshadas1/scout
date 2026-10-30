#!/bin/bash
#SBATCH --job-name=scout-E1c-eval
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/E1c-eval-%j.out

set -e
cd ~/scout

module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

python training/eval.py \
    --model results/E1c/weights/best.pt \
    --data ~/scratch/scout_data \
    --split val \
    --n 500 \
    --seed 4605 \
    --imgsz 640 \
    --thr 0.25 \
    --name E1c_eval \
    --batch 16
