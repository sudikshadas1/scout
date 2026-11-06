#!/bin/bash
# Runs any one command as a PACE job, so data tools and video tests stay off the login node.
#   sbatch training/slurm/run.sbatch python data/make_grayscale.py --data ~/scratch/scout_data
#SBATCH --job-name=scout-run
#SBATCH --account=ece
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd ~/scout
mkdir -p logs
module load anaconda3
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate scout

"$@"
