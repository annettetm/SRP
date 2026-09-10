#!/usr/bin/bash
#SBATCH --job-name=microlens
#SBATCH --output=logs/microlens%j.log
#SBATCH --error=logs/microlens%j.err
#SBATCH --mail-user=annettemathew2002@gmail.com
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1

cd /home/mathew/SRP/microlens
source ~/miniconda3/etc/profile.d/conda.sh
conda activate beauty
srun python scripts/read_emb.py "$@"
echo "The batch script ends."
