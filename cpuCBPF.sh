#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=normal
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# load the environment
source activate cbpfnet

python main.py
 
