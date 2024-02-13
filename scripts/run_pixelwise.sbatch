#!/bin/bash
#SBATCH -J rcf_pixelwise      # job name
#SBATCH -o model.out          # standard output file
#SBATCH -e model.err          # standard error file
#SBATCH --gres=gpu:1          # number of GPUs
#SBATCH --ntasks-per-node 4   # number of cores
#SBATCH -p gpu_test           # partition
#SBATCH -t 0-01:00            # time in D-HH:MM
#SBATCH --mem=32G              # memory in GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kyan@g.harvard.edu

# Activate conda env
eval "$(/n/home07/kayan/miniconda3/bin/conda shell.bash hook)"
conda activate geo-ml

# load modules
module load cuda/12.2.0-fasrc01

# Run program
srun -c 1  python3 asm_run_rcf_pixelwise.py
