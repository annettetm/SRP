#!/bin/bash
#SBATCH --job-name=proxyrca_tune_gated
#SBATCH --output=logs/proxyrca_tune_gated_ID%j.log
#SBATCH --error=logs/proxyrca_tune_gated_ID%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCA1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python tune1.py gated_ID/proxyrca --n_trials 30 --trial_timeout 10800
echo "The batch script ends."