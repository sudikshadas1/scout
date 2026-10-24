#!/bin/bash
# Evaluate one trained run and explain its misses: eval.py, then analyze_errors.py on the same 500 val images.
# Purpose: one command per experiment gives the checkpoint numbers plus the failure breakdown.
#   sbatch training/slurm/eval_pace.sbatch E1_full     # weights at results/E1_full/weights/best.pt
#   sbatch training/slurm/eval_pace.sbatch E1b
#SBATCH --job-name=scout-eval
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
NAME=${1:?usage: sbatch eval_pace.sbatch <run-name> [n-images]}
N=${2:-500}
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

W=results/$NAME/weights/best.pt
python training/eval.py --model "$W" --data ~/scratch/scout_data --split val --n "$N" --name "${NAME}_eval" --batch 16
python training/analyze_errors.py --model "$W" --data ~/scratch/scout_data --split val --n "$N" \
    --name "${NAME}_analysis" --thr 0.25
python training/analyze_errors.py --model "$W" --data ~/scratch/scout_data --split val --n "$N" \
    --name "${NAME}_analysis_t05" --thr 0.05
python training/vote_sim.py sweep "results/${NAME}_eval/threshold_sweep.csv"
