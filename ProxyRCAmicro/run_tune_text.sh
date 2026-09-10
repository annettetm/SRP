#!/bin/bash
#SBATCH --job-name=proxyrca_tune_text
#SBATCH --output=logs/proxyrca_tune_text_%j.log
#SBATCH --error=logs/proxyrca_tune_text_%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCAmicro
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python tune.py micro_text/proxyrca --n_trials 20 --trial_timeout 5400
echo "The batch script ends."
