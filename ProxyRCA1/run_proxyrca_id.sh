#!/bin/bash
#SBATCH --job-name=id_beauty
#SBATCH --output=logs/id_beauty%j.log
#SBATCH --error=logs/id_beauty%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1


#source activate proxyrca
#srun python entry.py beauty1/proxyrca

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCA1                    # navigate to the directory if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python entry.py beauty1_id/proxyrca  #  python jobs require the srun command to work
echo "The batch script ends."