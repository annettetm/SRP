#!/bin/bash
#SBATCH --job-name=check_id_scores
#SBATCH --output=logs/check_id_scores%j.log
#SBATCH --error=logs/check_id_scores%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCA1                    # navigate to the directory if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python check_id_scores.py beauty1_id/proxyrca --num_users 15   #  python jobs require the srun command to work
echo "The batch script ends."
