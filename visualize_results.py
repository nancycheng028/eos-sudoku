#!/usr/bin/env python3
"""
Visualize EoS runs saved under experiments/**/checkpoint_seed_0.pth

Outputs:
  experiments/comparison_loss.png
  experiments/comparison_acc.png
  experiments/comparison_sharpness.png
  experiments/phase_ratio.png          (sharpness / (2/lr) over steps)
  experiments/summary.csv              (last metrics per run)

Run headless-safe:
  export MPLBACKEND=Agg
  python visualize_results.py
"""
from __future__ import annotations
import math
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
OUT_ROOT = EXP_ROOT  # write comparison plots into experiments/


def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def find_runs(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("*/checkpoint_seed_0.pth") if p.is_file()])


def load_run(ckpt_path: Path) -> Dict:
    # safer load to avoid pickle execution warning in future PyTorch versions
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # torch >= 2.4
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")  # fallback for older torch
    hist = ckpt.get("history", {})
    args = ckpt.get("args", {})
    run = {
        "name": ckpt_path.parent.name,
        "path": ckpt_path,
        "args": args,
        "eos_threshold": np.array(hist.get("eos_threshold", []), dtype=float) if "eos_threshold" in hist.keys() else float(ckpt.get("eos_threshold", float("nan"))),
        "lrs": np.array(hist.get("lrs", []), dtype=float),
        "step": np.array(hist.get("step", []), dtype=float),
        "loss": np.array(hist.get("loss", []), dtype=float),
        "acc": np.array(hist.get("acc", []), dtype=float),
        "grad_norm": np.array(hist.get("grad_norm", []), dtype=float),
        "sharpness": np.array(hist.get("sharpness", []), dtype=float),
    }
    return run


def plot_comparison(runs: List[Dict], key: str, ylabel: str, out_name: str):
    plt.figure(figsize=(9, 5))
    any_curve = False
    for r in runs:
        x = r["step"]
        y = r.get(key)
        if x.size == 0 or y.size == 0:
            continue
        m = _finite(x) & _finite(y)
        if m.any():
            plt.plot(x[m], y[m], label=r["name"])
            any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out = OUT_ROOT / out_name
    plt.savefig(out, dpi=150)
    print(f"[viz] wrote {out}")


def plot_sharpness_with_threshold(runs: List[Dict]):
    plt.figure(figsize=(10, 6))
    any_curve = False
    for r in runs:
        # this is a very hacky way to fix but i wanted it to be compatible with the old output files as well where EoS thresholds were floats not arrays
        x, s, thr = r["step"], r["sharpness"], r["eos_threshold"] if type(r["eos_threshold"]) != float else np.full(len(r["step"]), r["eos_threshold"])
        if x.size == 0 or s.size == 0:
            continue
        m = _finite(x) & _finite(s)
        if m.any():
            plt.plot(x[m], s[m], label=f"{r['name']} sharp")
            plt.plot(x[m], thr[m], linestyle='--', label=f"2/lr {r['name']}")
            any_curve = True
            # thr = r.get("eos_threshold", float("nan"))
            # if not math.isnan(thr):
            #     plt.hlines(thr, xmin=x[m].min(), xmax=x[m].max(), linestyles="--", label=f"2/lr {r['name']}")
    if not any_curve:
        plt.close()
        return
    plt.xlabel("step")
    plt.ylabel("sharpness (≈ λ_max)")
    plt.legend(ncol=2)
    plt.tight_layout()
    out = OUT_ROOT / "comparison_sharpness.png"
    plt.savefig(out, dpi=150)
    print(f"[viz] wrote {out}")


def plot_phase_ratio(runs: List[Dict]):
    """Plot sharpness / (2/lr) over steps; >1 implies EoS exceed."""
    plt.figure(figsize=(9, 5))
    any_curve = False
    for r in runs:
        x, s, thr = r["step"], r["sharpness"], r["eos_threshold"] if type(r["eos_threshold"]) != float else np.full(len(r["step"]), r["eos_threshold"])
        #thr = r.get("eos_threshold", float("nan"))
        if x.size == 0 or s.size == 0 or thr.size == 0: #or math.isnan(thr) or thr == 0:
            continue
        ratio = s / thr
        m = _finite(x) & _finite(ratio)
        if m.any():
            plt.plot(x[m], ratio[m], label=r["name"])
            any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.axhline(1.0, linestyle=":", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("sharpness / (2/lr)")
    plt.legend()
    plt.tight_layout()
    out = OUT_ROOT / "phase_ratio.png"
    plt.savefig(out, dpi=150)
    print(f"[viz] wrote {out}")


def summarize(runs: List[Dict]):
    rows = []
    for r in runs:
        step = r["step"]
        last = -1 if step.size else None
        max_sharp = float(np.nanmax(r["sharpness"])) if r["sharpness"].size else float("nan")
        #thr = float(r.get("eos_threshold", float("nan")))
        rows.append({
            "run": r["name"],
            "steps": int(step.size),
            "last_step": int(step[last]) if step.size else -1,
            "final_loss": float(r["loss"][last]) if r["loss"].size else float("nan"),
            "final_acc": float(r["acc"][last]) if r["acc"].size else float("nan"),
            "max_sharpness": max_sharp,
            #"eos_threshold": thr,
            #"exceeded_eos": (max_sharp > thr) if (not math.isnan(thr)) else False,
        })
    df = pd.DataFrame(rows)
    out_csv = OUT_ROOT / "summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[viz] wrote {out_csv}")
    # pretty print
    with pd.option_context('display.max_colwidth', None):
        print("\n=== Summary (last metrics) ===")
        print(df.to_string(index=False))


def main():
    EXP_ROOT.mkdir(exist_ok=True)
    runs = [load_run(p) for p in find_runs(EXP_ROOT)]
    if not runs:
        print(f"[viz] no runs found under {EXP_ROOT}/**/checkpoint_seed_0.pth")
        return

    plot_comparison(runs, key="loss", ylabel="train loss", out_name="comparison_loss.png")
    plot_comparison(runs, key="acc", ylabel="train acc (masked)", out_name="comparison_acc.png")
    plot_sharpness_with_threshold(runs)
    plot_phase_ratio(runs)
    summarize(runs)


if __name__ == "__main__":
    main()
