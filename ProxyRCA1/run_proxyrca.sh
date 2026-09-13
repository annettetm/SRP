#!/bin/bash
#SBATCH --job-name=beauty1_id
#SBATCH --output=logs/beauty1_id%j.log
#SBATCH --error=logs/beauty1_id%j.err
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
#srun python -m py_compile models/carca.py models/carca_gated.py solvers/carca_gated.py
srun python entry.py beauty1_id/proxyrca  #  python jobs require the srun command to work
echo "The batch script ends."