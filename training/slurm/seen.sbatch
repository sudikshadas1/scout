#!/bin/bash
# Seen-location score (plan section 7): build the seen split once, then score a run on it.
#   sbatch training/slurm/seen.sbatch E1_full
#SBATCH --job-name=scout-seen
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout
NAME=${1:-E1_full}
if [ ! -d ~/scratch/scout_data/yolo/images/seen ]; then
    python data/build_seen_split.py --data ~/scratch/scout_data
fi
python training/eval.py --model results/$NAME/weights/best.pt --data ~/scratch/scout_data --split seen \
    --n 500 --name "${NAME}_seen" --batch 16
echo "compare results/${NAME}_seen/metrics.json (seen locations) with results/${NAME}_eval/metrics.json (new locations)"
