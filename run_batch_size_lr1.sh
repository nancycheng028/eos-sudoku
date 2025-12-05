#!/usr/bin/env bash
set -e
export MPLBACKEND=Agg
cd "$(dirname "$0")"

run() {
  echo ">>> $*"
  python sudoku_eos_baseline.py "$@"
}

# ---------- SGD, constant LR, change batch size ----------
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 1 --epochs 100  --batch_size 32 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 1 --epochs 100  --batch_size 64 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 1 --epochs 100  --batch_size 128 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 1 --epochs 100  --batch_size 256 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 1 --epochs 100  --batch_size 512 --max_samples 500000
run --csv "sudoku_anne_sl.csv" --optim sgd --lr 1 --epochs 100  --batch_size 1024 --max_samples 500000