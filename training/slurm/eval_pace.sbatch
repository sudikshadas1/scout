#!/bin/bash
# Evaluate one trained run and explain its misses: eval.py, then analyze_errors.py on the same 500 val images.
# Purpose: one command per experiment gives the checkpoint numbers plus the failure breakdown.
#   sbatch training/slurm/eval_pace.sbatch E1_full     # weights at results/E1_full/weights/best.pt
#   sbatch training/slurm/eval_pace.sbatch E1b
#   sbatch training/slurm/eval_pace.sbatch E1b 500 416 # at the Pi's picture size -> results/E1b_eval416, E1b_analysis416
#SBATCH --job-name=scout-eval
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
NAME=${1:?usage: sbatch eval_pace.sbatch <run-name> [n-images] [imgsz]}
N=${2:-500}
IMGSZ=${3:-640}
TAG=""
if [ "$IMGSZ" != "640" ]; then TAG=$IMGSZ; fi
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

W=results/$NAME/weights/best.pt
python training/eval.py --model "$W" --data ~/scratch/scout_data --split val --n "$N" --imgsz "$IMGSZ" \
    --name "${NAME}_eval${TAG}" --batch 16
python training/analyze_errors.py --model "$W" --data ~/scratch/scout_data --split val --n "$N" --imgsz "$IMGSZ" \
    --name "${NAME}_analysis${TAG}" --thr 0.25
python training/analyze_errors.py --model "$W" --data ~/scratch/scout_data --split val --n "$N" --imgsz "$IMGSZ" \
    --name "${NAME}_analysis${TAG}_t05" --thr 0.05
python training/vote_sim.py sweep "results/${NAME}_eval${TAG}/threshold_sweep.csv"
