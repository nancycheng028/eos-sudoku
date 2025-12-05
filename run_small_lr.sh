#!/usr/bin/env bash
set -e
export MPLBACKEND=Agg
cd "$(dirname "$0")"

run() {
  echo ">>> $*"
  python sudoku_eos_baseline.py "$@"
}

# ---------- SGD, constant LR ----------
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 0.00001 --epochs 100  --batch_size 512 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 0.0001 --epochs 100  --batch_size 512 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 0.001 --epochs 100  --batch_size 512 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 0.005 --epochs 100  --batch_size 512 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 0.01 --epochs 100  --batch_size 512 --max_samples 500000