import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
EPOCHS, BATCH, LR = 12, 1024, 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="шлях до локального creditcard.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    assert len(df) == 284_807 and int(df["Class"].sum()) == 492, "Неочікуваний набір даних"
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["Class"].to_numpy(dtype=np.int64)
    amount_raw = df["Amount"].to_numpy(dtype=np.float64)

    idx = np.arange(len(df))
    train_idx, hold_idx = train_test_split(idx, test_size=0.4, stratify=y, random_state=0)
    val_idx, test_idx = train_test_split(hold_idx, test_size=0.5, stratify=y[hold_idx], random_state=0)

    split_stats = {}
    for name, ii in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        n, f = int(len(ii)), int(y[ii].sum())
        split_stats[name] = {"n": n, "fraud": f, "fraud_rate": f / n}
        print(f"{name:10s} n={n:6d} fraud={f:4d} rate={f / n:.6%}")

    scaler = StandardScaler().fit(X[train_idx])
    Xtr = scaler.transform(X[train_idx]).astype(np.float32)
    Xva = scaler.transform(X[val_idx]).astype(np.float32)
    Xte = scaler.transform(X[test_idx]).astype(np.float32)
    ytr = y[train_idx].astype(np.float32)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(29, 32), nn.ReLU(), nn.Linear(32, 1))
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    n = len(Xtr)
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, BATCH): 
            b = torch.from_numpy(perm[start:start + BATCH])
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]).squeeze(1), ytr_t[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        log.append({"epoch": epoch, "train_loss": total / n})
        print(f"epoch {epoch:2d}  train loss {total / n:.6f}")

    model.eval()
    with torch.no_grad():
        z_val = model(torch.from_numpy(Xva)).squeeze(1)
        z_te = model(torch.from_numpy(Xte)).squeeze(1)
        s_val, s_te = torch.sigmoid(z_val), torch.sigmoid(z_te)

    torch.save(model.state_dict(), out / "model.pt")
    np.savez(out / "split_indices.npz", train=train_idx, validation=val_idx, test=test_idx)
    np.savez(
        out / "predictions.npz",
        val_row=val_idx, test_row=test_idx,
        y_val=y[val_idx], y_test=y[test_idx],
        z_val=z_val.numpy(), z_test=z_te.numpy(),
        s_val=s_val.numpy(), s_test=s_te.numpy(),        
        amount_val=amount_raw[val_idx], amount_test=amount_raw[test_idx],
    )
    (out / "scaler_stats.json").write_text(json.dumps(
        {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "features": FEATURES}, indent=1))
    (out / "split_stats.json").write_text(json.dumps(split_stats, indent=1))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"Збережено у {out}/")


if __name__ == "__main__":
    main()
