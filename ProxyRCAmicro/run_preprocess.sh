#!/bin/bash
#SBATCH --job-name=text1_preprocess
#SBATCH --output=logs/text1_preprocess%j.log
#SBATCH --error=logs/text1_preprocess%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

#source activate proxyrca
#srun python entry.py beauty1/proxyrca

echo "This is a test echo"
cd /home/mathew/SRP/ProxyRCAmicro                 # navigate to the directory if necessary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate proxyrca
srun python carca_preprocess1.py    #  python jobs require the srun command to work
echo "The batch script ends."

