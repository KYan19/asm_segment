#!/bin/bash
#SBATCH -J rasterize          # job name
#SBATCH -o rasterize.out      # standard output file
#SBATCH -e rasterize.err      # standard error file
#SBATCH -N 1                  # number of nodes
#SBATCH -c 1                  # number of cores
#SBATCH -p shared             # partition
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH --mem=8G              # memory in GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kyan@g.harvard.edu

# Activate conda env
eval "$(/n/home07/kayan/miniconda3/bin/conda shell.bash hook)"
conda activate geo

# Run program
python3 create_rasters.py
