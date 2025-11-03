import os, random, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class DummySudoku(Dataset):
    def __init__(self, n=20000):
        self.X = torch.randint(0, 10, (n, 81))      # 81 cells, digits 0..9 (0 = empty)
        self.y = torch.randint(1, 10, (n, 81))      # pretend solution digits 1..9
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = torch.nn.functional.one_hot(self.X[i], num_classes=10).float()  # (81,10)
        y = self.y[i] - 1  # targets 0..8
        return x, y

class TinyRecurrentBlock(nn.Module):
    def __init__(self, d_model=128, steps=4):
        super().__init__()
        self.steps = steps
        self.inp = nn.Linear(10, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, 9)  # logits for digits 1..9
    def forward(self, x):                   # x: (B,81,10)
        h = torch.relu(self.inp(x))
        for _ in range(self.steps):
            h, _ = self.gru(h)             # message passing-ish refinement
        return self.head(h)                # (B,81,9)

def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    device = "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available()) else "cpu"
    ds = DummySudoku(n=max(cfg["dataset"]["train_limit"], 2000))
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)

    model = TinyRecurrentBlock(d_model=cfg["model"]["d_model"], steps=cfg["model"]["n_steps"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    crit = nn.CrossEntropyLoss()

    os.makedirs("logs", exist_ok=True)
    rows, step = [], 0
    for epoch in range(cfg["training"]["epochs"]):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)                            # (B,81,9)
            loss = crit(logits.reshape(-1, 9), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            if cfg["training"]["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            opt.step()

            if step % cfg["training"]["log_every"] == 0:
                rows.append({"step": step, "loss": float(loss.item())})
                print(f"[step {step:6d}] loss={loss.item():.4f}", flush=True)
            step += 1

    pd.DataFrame(rows).to_csv(cfg["training"]["out_csv"], index=False)
    print(f"[done] wrote {cfg['training']['out_csv']}")
if __name__ == "__main__":
    main()
