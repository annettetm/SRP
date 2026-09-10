#!/bin/bash
#SBATCH --job-name=preprocess_fashion
#SBATCH --output=logs/%x_%j.log 
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=lakshmisureshbabu095@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1

source activate proxyrca
srun python preprocess.py prepare --dname fashion
srun python preprocess.py split_quarters --dname fashion
srun python preprocess.py count_stats


