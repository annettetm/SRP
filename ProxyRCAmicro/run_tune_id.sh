#!/bin/bash
#SBATCH --job-name=proxyrca_tune_id
#SBATCH --output=logs/proxyrca_tune_id_%j.log
#SBATCH --error=logs/proxyrca_tune_id_%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCAmicro
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python tune.py micro_id/proxyrca --n_trials 20
echo "The batch script ends."
