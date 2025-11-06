#!/bin/bash
#SBATCH -J sudoku-baseline
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o logs/baseline_%j.out
#SBATCH -e logs/baseline_%j.err

set -euo pipefail

# ---- Paths ----
PROJECT_DIR="/orcd/home/002/cheng028/eos-sudoku"
CSV_LINK="${PROJECT_DIR}/sudoku.csv"
cd "$PROJECT_DIR"

mkdir -p logs results experiments

# ---- Env (non-interactive, nounset-safe, no ~/.bashrc) ----
set +u
export BASHRCSOURCED=1
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  # fallback: Engaging miniforge path (adjust if your site differs)
  source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
fi
conda activate eos
set -u

# Headless plotting + perf
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCH_SHOW_CPP_STACKTRACES=1

# ---- Config knobs ----
EPOCHS=6
BATCH=512
LR=0.10
MAX_SAMPLES=60000
SEED=0

# Create run id + dirs
RUN_ID=$(date +"%Y%m%d_%H%M%S")
EXP="lr${LR}_bs${BATCH}_ep${EPOCHS}_seed${SEED}"
RESULTS_DIR="${PROJECT_DIR}/results/${RUN_ID}_${EXP}"
EXP_DIR="${PROJECT_DIR}/experiments/${EXP}"
mkdir -p "$RESULTS_DIR" "$EXP_DIR"

# ---- Sanity checks ----
python - <<'PY'
import torch, pathlib, os
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("CWD:", os.getcwd())
print("CSV exists:", pathlib.Path("sudoku.csv").exists())
PY

# ---- Baseline run ----
python -u sudoku_eos_baseline.py \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --lr "${LR}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

# Archive outputs for viz
mv -f sudoku_eos_checkpoint.pth "${EXP_DIR}/checkpoint_seed_0.pth"
mv -f eos_analysis.png          "${EXP_DIR}/eos_analysis.png"

# ---- Generate comparison/phase plots ----
python -u visualize_results.py

# Collect all PNGs for this run
find "${PROJECT_DIR}" -maxdepth 3 -type f -name "*.png" -print0 | \
  xargs -0 -I {} bash -c 'base=$(basename "{}"); cp -f "{}" "'"${RESULTS_DIR}"'/$base"'

# Manifest
cat > "${RESULTS_DIR}/run_manifest.txt" <<EOF
Run ID      : ${RUN_ID}
Experiment  : ${EXP}
Timestamp   : $(date -Is)
Slurm JobID : ${SLURM_JOB_ID:-N/A}
Params      : epochs=${EPOCHS}, batch=${BATCH}, lr=${LR}, max_samples=${MAX_SAMPLES}, seed=${SEED}
CSV         : ${CSV_LINK}
Project Dir : ${PROJECT_DIR}
Results Dir : ${RESULTS_DIR}
EOF

echo "[done] Plots and artifacts copied to: ${RESULTS_DIR}"
