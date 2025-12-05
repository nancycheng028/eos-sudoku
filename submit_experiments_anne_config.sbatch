#!/bin/bash
#SBATCH -J sudoku_test_batch
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 05:00:00
#SBATCH -p mit_normal_gpu
#SBATCH -o logs/sudoku_test_batch_%j.out
#SBATCH -e logs/sudoku_test_batch_%j.err

# load your environment
module load miniforge
conda activate eos-sudoku-env

cd /home/annegv/eos-sudoku

# make sure logs dir exists
mkdir -p logs
./run_small_lr.sh