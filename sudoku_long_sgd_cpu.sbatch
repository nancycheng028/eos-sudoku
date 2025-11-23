#!/bin/bash
#SBATCH -J sudoku_long_sgd
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -p mit_normal
#SBATCH -o logs/sudoku_long_sgd_%j.out
#SBATCH -e logs/sudoku_long_sgd_%j.err

# load your environment
source ~/.bashrc
conda activate eos

cd /orcd/home/002/cheng028/eos-sudoku

# make sure logs dir exists
mkdir -p logs

# run the longer-epoch experiments (0.1, 0.5, 1.0, step schedules)
./run_longer_epoch_experiments.sh

