#!/bin/bash
# E8: the cat / not-cat crop check on E1b's detections at the trap's picture size.
# Needs results/E1b_pipeline416/detections.jsonl (sbatch training/slurm/eval_pace.sbatch E1b 500 416).
# Uses Wellington crops too if e7_data.sbatch has run.
# If compute nodes have no internet, download the ImageNet weights once on the login node first:
#   python -c "from torchvision import models; models.mobilenet_v3_small(weights='IMAGENET1K_V1')"
#   sbatch training/slurm/e8_crop.sbatch
#SBATCH --job-name=scout-E8
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

DATA=~/scratch/scout_data
EXTRA=""
if [ -d "$DATA/wct/yolo/images/train" ]; then EXTRA="--extra-data $DATA/wct"; fi
python training/train_crop_classifier.py --data "$DATA" $EXTRA --workers 8 \
    --detections results/E1b_pipeline416/detections.jsonl --name E8
