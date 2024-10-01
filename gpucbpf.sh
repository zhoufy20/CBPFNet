#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# load the environment
source activate cbpfnetgpu

python main.py
 

