#!/usr/bin/env python3
"""
Visualization for Sudoku EoS experiments.

Scans experiments/**/checkpoint_seed_*.pth and generates:
  - experiments/comparison_loss.png
  - experiments/comparison_acc.png
  - experiments/comparison_sharpness.png
  - experiments/phase_ratio.png
  - experiments/summary.csv

Usage:
    export MPLBACKEND=Agg
    python visualize_results.py               # all runs
    python visualize_results.py --include adam_    # only Adam runs
    python visualize_results.py --exclude adam_    # all non-Adam runs
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import matplotlib
if not matplotlib.get_backend().lower().startswith("agg"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_ROOT = Path("experiments")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize EoS experiments")
    p.add_argument(
        "--include", type=str, default="",
        help="comma-sep substrings; keep runs whose name contains ANY of these (e.g. 'adam_')."
    )
    p.add_argument(
        "--exclude", type=str, default="",
        help="comma-sep substrings; drop runs whose name contains ANY of these."
    )
    return p.parse_args()


def _name_matches(name: str, include: str, exclude: str) -> bool:
    if include:
        inc = [s for s in include.split(",") if s]
        if not any(s in name for s in inc):
            return False
    if exclude:
        exc = [s for s in exclude.split(",") if s]
        if any(s in name for s in exc):
            return False
    return True


def find_runs(root: Path) -> List[Path]:
    return sorted(root.rglob("checkpoint_seed_*.pth"))


def load_run(ckpt_path: Path) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hist = ckpt["history"]
    run_name = ckpt_path.parent.name
    return {
        "name": run_name,
        "path": ckpt_path,
        "history": hist,
    }


def plot_comparison(runs: List[Dict], key: str, ylabel: str, out_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        hist = run["history"]
        steps = np.array(hist["step"], dtype=int)
        values = np.array(hist[key], dtype=float)
        m = np.isfinite(values)
        if not m.any():
            continue
        ax.plot(steps[m], values[m], label=run["name"])
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = EXP_ROOT / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def plot_sharpness_with_threshold(runs: List[Dict]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        hist = run["history"]
        steps = np.array(hist["step"], dtype=int)
        sharp = np.array(hist["sharpness"], dtype=float)
        m = np.isfinite(sharp)
        if not m.any():
            continue
        ax.plot(steps[m], sharp[m], label=run["name"])
    ax.set_xlabel("step")
    ax.set_ylabel("sharpness (λ_max)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = EXP_ROOT / "comparison_sharpness.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def plot_phase_ratio(runs: List[Dict]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        hist = run["history"]
        steps = np.array(hist["step"], dtype=int)
        sharp = np.array(hist["sharpness"], dtype=float)
        thr = np.array(hist["eos_threshold"], dtype=float)
        m = np.isfinite(sharp) & np.isfinite(thr) & (thr != 0.0)
        if not m.any():
            continue
        ratio = sharp[m] / thr[m]
        ax.plot(steps[m], ratio, label=run["name"])
    ax.axhline(1.0, color="gray", linestyle="--", label="EoS = 1")
    ax.set_xlabel("step")
    ax.set_ylabel("sharpness / (2/lr)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path = EXP_ROOT / "phase_ratio.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


def summarize(runs: List[Dict]):
    rows = []
    for run in runs:
        name = run["name"]
        hist = run["history"]
        steps = np.array(hist["step"], dtype=int)
        loss = np.array(hist["loss"], dtype=float)
        acc = np.array(hist["acc"], dtype=float)
        sharp = np.array(hist["sharpness"], dtype=float)
        thr = np.array(hist["eos_threshold"], dtype=float)

        n_sharp = np.isfinite(sharp).sum()
        print(f"{name} sharpness {n_sharp} entries of {len(sharp)} in sharpness history are non-nan")

        m_ratio = np.isfinite(sharp) & np.isfinite(thr) & (thr != 0.0)
        n_ratio = m_ratio.sum()
        print(f"{name} ratio {n_ratio} entries of {len(sharp)} in ratio calculation are non-nan")

        last_step = int(steps[-1]) if len(steps) else -1
        final_loss = float(loss[-1]) if len(loss) else float("nan")
        final_acc = float(acc[-1]) if len(acc) else float("nan")
        max_sharp = float(np.nanmax(sharp)) if np.isfinite(sharp).any() else float("nan")

        rows.append({
            "run": name,
            "steps": len(steps),
            "last_step": last_step,
            "final_loss": final_loss,
            "final_acc": final_acc,
            "max_sharpness": max_sharp,
        })

    df = pd.DataFrame(rows, columns=["run", "steps", "last_step", "final_loss", "final_acc", "max_sharpness"])
    out_path = EXP_ROOT / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[viz] wrote {out_path}")

    print("\n=== Summary (last metrics) ===")
    print(df.to_string(index=False))


def main():
    args = parse_args()

    ckpts = find_runs(EXP_ROOT)
    if not ckpts:
        print(f"[viz] no runs found under {EXP_ROOT}/**/checkpoint_seed_0.pth")
        return

    runs = [load_run(p) for p in ckpts]
    runs = [r for r in runs if _name_matches(r["name"], args.include, args.exclude)]

    if not runs:
        print("[viz] no runs matched the include/exclude filters")
        return

    plot_comparison(runs, key="loss", ylabel="train loss", out_name="comparison_loss.png")
    plot_comparison(runs, key="acc", ylabel="train acc (masked)", out_name="comparison_acc.png")
    plot_sharpness_with_threshold(runs)
    plot_phase_ratio(runs)
    summarize(runs)


if __name__ == "__main__":
    main()
