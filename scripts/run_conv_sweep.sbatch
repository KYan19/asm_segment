#!/bin/bash
#SBATCH -J conv_sweep         # job name
#SBATCH -o model.out          # standard output file
#SBATCH -e model.err          # standard error file
#SBATCH --gres=gpu:1          # number of GPUs
#SBATCH -p gpu_test           # partition
#SBATCH -t 0-00:20            # time in D-HH:MM
#SBATCH --mem=32G             # memory in GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kyan@g.harvard.edu

# Activate conda env
eval "$(/n/home07/kayan/miniconda3/bin/conda shell.bash hook)"
conda activate geo-ml

# load modules
module load cuda/12.2.0-fasrc01

# Run program
python3 asm_run_conv_sweep.py
