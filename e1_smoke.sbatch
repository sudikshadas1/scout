#!/bin/bash
#SBATCH --job-name=scout-E1-smoke
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/E1-smoke-%j.out

set -e
cd ~/scout

module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

yolo detect train \
    model=yolo26n.pt \
    data=data/scout_pace.yaml \
    epochs=1 \
    imgsz=640 \
    batch=16 \
    workers=4 \
    device=0 \
    project=results \
    name=E1_smoke \
    plots=True \
    exist_ok=True
