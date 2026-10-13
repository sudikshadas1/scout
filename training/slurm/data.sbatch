#!/bin/bash
# CPU job: fetch metadata, count, build manifest, download images. No GPU needed.
# Run splits.py once by hand first and commit data/splits.json.
#   sbatch training/slurm/data.sbatch
#SBATCH -J scout-data
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH -o logs/data-%j.out

set -euo pipefail
DATA=${DATA:-$HOME/scratch/scout_data}
mkdir -p logs
module load anaconda3
source activate scout || conda activate scout
python data/fetch_metadata.py --out "$DATA"
python data/stats.py --data "$DATA"
python data/build_manifest.py --data "$DATA"
python data/download_images.py --data "$DATA" --workers 16
