#!/usr/bin/env bash
set -e
export MPLBACKEND=Agg
cd "$(dirname "$0")"

run() {
  echo ">>> $*"
  python sudoku_eos_baseline.py "$@"
}

# ---------- SGD, constant LR ----------
run --optim sgd --lr 0.10 --epochs 6  --batch-size 512 --run-name lr0.10_bs512_ep6_seed0
run --optim sgd --lr 0.10 --epochs 8  --batch-size 512 --run-name lr0.10_bs512_ep8_seed0
run --optim sgd --lr 0.5  --epochs 8  --batch-size 512 --run-name lr0.5_bs512_ep8_ms50k_seed0
run --optim sgd --lr 1.0  --epochs 8  --batch-size 512 --run-name lr1.0_bs512_ep8_ms50k_seed0
run --optim sgd --lr 2.0  --epochs 8  --batch-size 512 --run-name lr2.0_bs512_ep8_ms50k_seed0
run --optim sgd --lr 5.0  --epochs 8  --batch-size 512 --run-name lr5.0_bs512_ep8_ms50k_seed0
run --optim sgd --lr 10.0 --epochs 20 --batch-size 512 --run-name lr10.0_ep20_ms50k_seed0

# ---------- SGD, step LR ----------
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 0.5  --epochs 8 --batch-size 512 --run-name lr1.0_step_g0.5_bs512_ep8_seed0
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 0.75 --epochs 8 --batch-size 512 --run-name lr1.0_step_g0.75_bs512_ep8_seed0
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 0.9  --epochs 8 --batch-size 512 --run-name lr1.0_step_g0.9_bs512_ep8_seed0
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 0.95 --epochs 8 --batch-size 512 --run-name lr1.0_step_g0.95_bs512_ep8_seed0
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 1.0  --epochs 8 --batch-size 512 --run-name lr1.0_step_g1.0_bs512_ep8_seed0

# ---------- Adam ----------
run --optim adam --lr 0.0003 --epochs 8 --batch-size 512 --run-name adam_lr0.0003_bs512_ep8_seed0
run --optim adam --lr 0.003  --epochs 8 --batch-size 512 --run-name adam_lr0.003_bs512_ep8_seed0
run --optim adam --lr 0.01   --epochs 8 --batch-size 512 --run-name adam_lr0.01_bs512_ep8_seed0
run --optim adam --lr 0.03   --epochs 8 --batch-size 512 --run-name adam_lr0.03_bs512_ep8_seed0

