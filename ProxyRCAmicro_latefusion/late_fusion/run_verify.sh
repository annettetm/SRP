#!/bin/bash
#SBATCH --job-name=micro_late_verify
#SBATCH --output=late_fusion/results/verify_%j.out
#SBATCH --error=late_fusion/results/verify_%j.err
#SBATCH --partition=STUD
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mathew/SRP/ProxyRCAmicro_latefusion
source /home/mathew/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca

export PYTHONDONTWRITEBYTECODE=1
srun python -B late_fusion/verify_models.py
