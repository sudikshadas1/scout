#!/bin/bash
# E2: MobileNetV3 cat / not-cat classifier (comparison only, plan sections 5 and 9).
#   sbatch training/slurm/e2.sbatch
# If compute nodes have no internet, download the ImageNet weights once on the login node first:
#   python -c "import torchvision as t; t.models.mobilenet_v3_small(weights='IMAGENET1K_V1')"
#SBATCH --job-name=scout-E2
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout
python training/train_classifier.py --data ~/scratch/scout_data --name E2 --epochs 12 --workers 4
