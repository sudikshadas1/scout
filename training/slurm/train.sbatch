#!/bin/bash
# Train one experiment on a PACE ICE GPU (wraps training/train_yolo.py).
# Purpose: every experiment in the plan (E1, E3, E5 ...) runs through this same job.
# Status:  Written, not yet run. Lines marked CHECK may need the GPU type and max walltime.
# Later:   Phase 2b: E5 passes --data-yaml data/scout_trap.yaml once that file exists.
# Pass the experiment name and any extra train_yolo.py flags:
#   sbatch training/slurm/train.sbatch E1
#   sbatch training/slurm/train.sbatch E5 --weights results/E3/weights/best.pt --epochs 50
#SBATCH -J scout-train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1            # CHECK: PACE ICE may want a GPU type, e.g. --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH -t 8:00:00              # CHECK: max walltime allowed for your course allocation
#SBATCH -o logs/train-%x-%j.out

set -euo pipefail
NAME=${1:?usage: sbatch train.sbatch <experiment-name> [extra args]}
shift
mkdir -p logs
module load anaconda3
source activate scout || conda activate scout
python training/train_yolo.py --name "$NAME" "$@"
