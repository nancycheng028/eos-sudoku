# src/train.py
import os, random, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from data.kaggle_csv import KaggleSudoku

class TinyRecurrentBlock(nn.Module):
    def __init__(self, d_model=128, steps=4, dropout=0.0):
        super().__init__()
        self.steps = steps
        self.inp = nn.Linear(10, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 9)  # logits for digits 1..9
    def forward(self, x):                   # x: (B,81,10)
        h = torch.relu(self.inp(x))
        for _ in range(self.steps):
            h, _ = self.gru(h)
            h = self.drop(h)
        return self.head(h)                 # (B,81,9)

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    # ---- Data ----
    file = os.path.expandvars(cfg["dataset"]["file"])
    subset = cfg["dataset"]["subset_rows"]
    ds = KaggleSudoku(
        csv_file=file,
        subset_rows=subset,
        puzzle_col=cfg["dataset"]["cols"]["puzzle"],
        solution_col=cfg["dataset"]["cols"]["solution"],
    )

    num_workers = int(cfg["dataset"].get("num_workers", 0))
    pin = bool(cfg["dataset"].get("pin_memory", False))
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                    shuffle=True, drop_last=True,
                    num_workers=num_workers, pin_memory=pin)

    # ---- Model/opt ----
    device = "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    model = TinyRecurrentBlock(
        d_model=cfg["model"]["d_model"],
        steps=cfg["model"]["n_steps"],
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    opt_name = cfg["optim"]["name"].lower()
    if opt_name != "adam":
        raise ValueError("Only Adam is configured in this baseline.")
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]), eps=cfg["optim"]["eps"],
        weight_decay=cfg["optim"]["weight_decay"]
    )
    crit = nn.CrossEntropyLoss()

    os.makedirs("logs", exist_ok=True)
    rows, step = [], 0

    # ---- Train ----
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        for x, y in dl:
            x, y = x.to(device), y.to(device)            # x:(B,81,10), y:(B,81)
            logits = model(x)                            # (B,81,9)
            loss = crit(logits.reshape(-1, 9), y.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip = cfg["optim"]["grad_clip"]
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            if step % cfg["training"]["log_every"] == 0:
                rows.append({"step": step, "loss": float(loss.item())})
                print(f"[step {step:6d}] loss={loss.item():.4f}", flush=True)
            step += 1

    pd.DataFrame(rows).to_csv(cfg["training"]["metrics_csv"], index=False)
    print(f"[done] wrote {cfg['training']['metrics_csv']}")
if __name__ == "__main__":
    main()