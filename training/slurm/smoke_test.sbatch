#!/bin/bash
# 3-epoch training on Ultralytics' tiny coco8 set: proves GPU + env + ultralytics work.
# Purpose: first PACE job; confirms the GPU request and conda env before any real training.
# Status:  Run on PACE: trained 3 epochs on an H100.
# Later:   nothing once it passes; keep it for checking the env after changes.
#   sbatch training/slurm/smoke_test.sbatch   then   cat logs/smoke-<jobid>.out
#SBATCH -J scout-smoke
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -o logs/smoke-%j.out

mkdir -p logs
module load anaconda3
source activate scout || conda activate scout
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available())"
yolo train model=yolo26n.pt data=coco8.yaml epochs=3 imgsz=640 project=results name=smoke exist_ok=True
