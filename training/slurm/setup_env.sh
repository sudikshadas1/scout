#!/bin/bash
# One-time setup on a PACE ICE login node: conda env + data folder on scratch.
#   bash training/slurm/setup_env.sh
#
# CHECK against the PACE ICE docs if a module name differs on your cluster ("module avail anaconda").
set -euo pipefail

module load anaconda3
ENV=scout
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.11
fi
source activate "$ENV" || conda activate "$ENV"
pip install -r "$(dirname "$0")/../../requirements.txt"

mkdir -p ~/scratch/scout_data
echo "env '$ENV' ready; data folder: ~/scratch/scout_data"
echo "next: sbatch training/slurm/smoke_test.sbatch"
