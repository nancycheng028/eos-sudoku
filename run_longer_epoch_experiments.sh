#!/usr/bin/env bash
set -e
export MPLBACKEND=Agg
cd "$(dirname "$0")"

run() {
  echo ">>> $*"
  python sudoku_eos_baseline.py "$@"
}

# ---------- Longer SGD, constant LR ----------

# lr = 0.10, 60epochs
run --optim sgd --lr 0.10 --epochs 60 \
    --batch-size 512 --max-samples 50000 \
    --run-name lr0.10_bs512_ep60_ms50k_seed0

# lr = 0.5, 16 epochs
run --optim sgd --lr 0.5 --epochs 16 \
    --batch-size 512 --max-samples 50000 \
    --run-name lr0.5_bs512_ep16_ms50k_seed0

# lr = 1.0, 16 epochs
run --optim sgd --lr 1.0 --epochs 16 \
    --batch-size 512 --max-samples 50000 \
    --run-name lr1.0_bs512_ep16_ms50k_seed0

# ---------- Longer SGD, step LR (representative schedules) ----------

# gamma = 0.5
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 0.5 \
    --epochs 16 --batch-size 512 --max-samples 50000 \
    --run-name lr1.0_step_g0.5_bs512_ep16_ms50k_seed0

# gamma = 0.9
run --optim sgd --lr 1.0 --lr_type step --lr_gamma 0.9 \
    --epochs 16 --batch-size 512 --max-samples 50000 \
    --run-name lr1.0_step_g0.9_bs512_ep16_ms50k_seed0
