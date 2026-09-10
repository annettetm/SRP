#!/bin/bash
#SBATCH --job-name=micro_late_fusion
#SBATCH --output=late_fusion/results/late_fusion_%j.out
#SBATCH --error=late_fusion/results/late_fusion_%j.err
#SBATCH --partition=STUD
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mathew/SRP/ProxyRCAmicro_latefusion
source /home/mathew/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca

export PYTHONDONTWRITEBYTECODE=1

srun python -B -c 'import ast, pathlib; path = pathlib.Path("late_fusion/evaluate_late_fusion.py"); ast.parse(path.read_text()); print("Python syntax valid")'
srun python -B late_fusion/evaluate_late_fusion.py