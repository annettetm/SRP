#!/bin/bash
#SBATCH --job-name=proxyrca_beauty1
#SBATCH --output=logs/proxyrca_text_%j.log
#SBATCH --error=logs/proxyrca_text_%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1

#source activate proxyrca
#srun python entry.py beauty1/proxyrca

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCAmicro                 # navigate to the directory if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python carca_preprocess.py    #  python jobs require the srun command to work
echo "The batch script ends."

