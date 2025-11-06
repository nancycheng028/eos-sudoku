#!/usr/bin/env python3
"""
Sudoku EoS Baseline (PyTorch)

- Loads Kaggle CSV (quizzes, solutions) expected at ./sudoku.csv
- Simple per-cell classifier (MLP) to fill blanks
- Logs loss, accuracy, grad-norm
- Approximates largest Hessian eigenvalue ("sharpness") via power iteration
  using autograd Hessian–vector products on a mini-batch
- Saves: sudoku_eos_checkpoint.pth (weights+history), eos_analysis.png (plot)

Run (headless safe):
    export MPLBACKEND=Agg
    python sudoku_eos_baseline.py --epochs 5 --lr 0.1 --max-samples 50000

Place this file at the project root:
    /orcd/home/002/cheng028/eos-sudoku/sudoku_eos_baseline.py

Tested with: torch 2.4.1+cu121, python 3.10
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
# for non-interactive environments, let user override via env
if not matplotlib.get_backend().lower().startswith("agg"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------
# Data utilities
# ------------------------------

def _str81_to_array(s: str) -> np.ndarray:
    s = s.strip()
    assert len(s) == 81, f"Expected 81 chars, got {len(s)}"
    return np.array([int(c) for c in s], dtype=np.int64)

class KaggleSudoku(Dataset):
    """Kaggle CSV with columns: 'quizzes', 'solutions'.

    Each item returns:
      x: [81, 11] features (10-d one-hot digit incl. 0 for blank, + 1 mask bit)
      y: [81] target digits in 1..9 (0 for blanks in labels is unused)
      mask: [81] boolean, True where we need to predict (original blanks)
    """
    def __init__(self, csv_path: Path, max_samples: int | None = None, seed: int = 0):
        df = pd.read_csv(csv_path)
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        self.quizzes = [_str81_to_array(s) for s in df["quizzes"].tolist()]
        self.solutions = [_str81_to_array(s) for s in df["solutions"].tolist()]

    def __len__(self) -> int:
        return len(self.quizzes)

    def __getitem__(self, idx: int):
        q = self.quizzes[idx]          # 0..9, 0 means blank
        sol = self.solutions[idx]      # 1..9 (complete solution)
        mask = (q == 0)                # predict only blanks
        # input encoding: per cell 11-d vector (10 one-hot for digit 0..9, + is_given flag)
        x = np.zeros((81, 11), dtype=np.float32)
        for i, d in enumerate(q):
            x[i, d] = 1.0  # one-hot where index 0 means blank
            x[i, 10] = 0.0 if d == 0 else 1.0  # is_given
        y = sol.astype(np.int64) - 1  # targets in 0..8 for CE
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask.astype(np.bool_))

# ------------------------------
# Model
# ------------------------------
class CellMLP(nn.Module):
    """MLP over flattened per-cell features.

    Input: [B, 81, 11] -> flatten -> MLP -> [B, 81*9] logits -> view [B, 81, 9]
    """
    def __init__(self, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(81 * 11, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 81 * 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        z = self.net(x.view(b, -1))
        return z.view(b, 81, 9)

# ------------------------------
# Training helpers
# ------------------------------
@dataclass
class History:
    step: List[int]
    loss: List[float]
    acc: List[float]
    grad_norm: List[float]
    sharpness: List[float]

    def as_dict(self):
        return {k: getattr(self, k) for k in ["step", "loss", "acc", "grad_norm", "sharpness"]}


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits: [B,81,9], targets: [B,81] in 0..8, mask: [B,81] True if originally blank
    B = logits.size(0)
    logits = logits[mask]
    targets = targets[mask]
    if logits.numel() == 0:
        return torch.zeros([], device=logits.device)
    return F.cross_entropy(logits, targets)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(1.0, device=logits.device)
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == targets[mask]).float()
    return correct.mean()


def grad_norm(parameters: List[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def _normalize(v: List[torch.Tensor]) -> float:
    sq = 0.0
    for t in v:
        sq += t.pow(2).sum().item()
    n = math.sqrt(max(sq, 1e-12))
    for t in v:
        t.div_(n)
    return n


def hessian_vector_product(loss: torch.Tensor, params: List[torch.nn.Parameter], v: List[torch.Tensor]) -> List[torch.Tensor]:
    # Compute Hessian-vector product using autograd: Hv = d/dp (grad(loss,p) · v)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    gv = 0.0
    for g, vi in zip(grads, v):
        gv = gv + (g * vi).sum()
    hv = torch.autograd.grad(gv, params, retain_graph=True)
    return [h.contiguous() if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]


def estimate_top_hessian_eig(model: nn.Module, loss_fn, batch, device: str, iters: int = 5) -> float:
    """Power iteration on Hessian to estimate lambda_max on a single batch.
    Keep this cheap: 3–5 iterations is fine for tracking trends.
    """
    model.zero_grad(set_to_none=True)
    x, y, m = (t.to(device) for t in batch)
    logits = model(x)
    loss = loss_fn(logits, y, m)
    params = [p for p in model.parameters() if p.requires_grad]
    # init v like params
    v = [torch.randn_like(p) for p in params]
    _normalize(v)
    lam = 0.0
    for _ in range(iters):
        Hv = hessian_vector_product(loss, params, v)
        lam = 0.0
        for hv in Hv:
            lam += (hv * hv).sum().item()
        lam = math.sqrt(lam)
        v = Hv
        _ = _normalize(v)
    return float(lam)

# ------------------------------
# Main train loop
# ------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    csv_path = Path(args.csv)
    assert csv_path.exists(), f"CSV not found: {csv_path}"
    ds = KaggleSudoku(csv_path, max_samples=args.max_samples, seed=args.seed)

    n_total = len(ds)
    n_val = max(1000, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device=="cuda"))

    model = CellMLP(hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)

    history = History(step=[], loss=[], acc=[], grad_norm=[], sharpness=[])
    step = 0
    eos_threshold = 2.0 / args.lr  # classical EoS threshold ~ 2 / eta

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x, y, m = (t.to(device) for t in batch)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = masked_ce_loss(logits, y, m)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            with torch.no_grad():
                acc = masked_accuracy(logits, y, m).item()
                gnorm = grad_norm(list(model.parameters()))

            # Cheap sharpness estimate every "log_every" steps using a small sub-batch
            sharp = float("nan")
            if step % args.log_every == 0:
                # sample a tiny batch from val for stability
                try:
                    batch_small = next(iter(val_loader))
                except StopIteration:
                    batch_small = batch
                sharp = estimate_top_hessian_eig(model, masked_ce_loss, batch_small, device, iters=args.sharpness_iters)

            history.step.append(step)
            history.loss.append(loss.item())
            history.acc.append(acc)
            history.grad_norm.append(gnorm)
            history.sharpness.append(sharp)

            if step % args.log_every == 0:
                print(f"[ep {epoch:02d} | step {step:06d}] loss={loss.item():.4f} acc={acc:.4f} "
                      f"grad={gnorm:.2f} sharp={sharp:.2f} thresh={eos_threshold:.2f}")
            step += 1

        # end epoch: quick val report
        model.eval()
        with torch.no_grad():
            v_loss, v_acc, v_n = 0.0, 0.0, 0
            for bx, by, bm in val_loader:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                logits = model(bx)
                v_loss += float(masked_ce_loss(logits, by, bm).item()) * bx.size(0)
                v_acc += float(masked_accuracy(logits, by, bm).item()) * bx.size(0)
                v_n += bx.size(0)
            v_loss /= max(v_n, 1)
            v_acc /= max(v_n, 1)
            print(f"[val ep {epoch:02d}] loss={v_loss:.4f} acc={v_acc:.4f}")

    # Save checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "history": history.as_dict(),
        "eos_threshold": eos_threshold,
    }
    out_path = Path("sudoku_eos_checkpoint.pth")
    torch.save(ckpt, out_path)
    print(f"[done] wrote {out_path.resolve()}")

    # Plot
    plot_history(history, eos_threshold)


def plot_history(history: History, eos_threshold: float, out_png: str = "eos_analysis.png"):
    steps = np.array(history.step)
    loss = np.array(history.loss)
    acc = np.array(history.acc)
    grad = np.array(history.grad_norm)
    sharp = np.array(history.sharpness)

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(steps, loss)
    axes[0].set_ylabel("train loss")

    axes[1].plot(steps, acc)
    axes[1].set_ylabel("train acc (masked)")

    axes[2].plot(steps, grad)
    axes[2].set_ylabel("grad-norm")

    axes[3].plot(steps, sharp, label="sharpness (≈ λ_max)")
    axes[3].axhline(eos_threshold, linestyle="--", label="2 / lr")
    axes[3].set_ylabel("sharpness")
    axes[3].set_xlabel("step")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[done] wrote {out_png}")


# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sudoku EoS Baseline")
    p.add_argument("--csv", type=str, default="sudoku.csv", help="Path to kaggle sudoku.csv")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=0.0, help="grad clipping max-norm; 0 disables")
    p.add_argument("--max-samples", type=int, default=50000, help="limit dataset for speed; None for all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sharpness-iters", type=int, default=4, help="power-iter steps for Hessian eig")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
