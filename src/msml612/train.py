"""Train the quantile PatchTST baseline.

  python -m msml612.train --config configs/baseline_cpu_small.yaml [--set train.epochs=2 ...]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler

from .data.windows import SplitSpec, WindowDataset, load_npz
from .metrics import pinball_loss
from .models.quantile_patchtst import QuantilePatchTST


def load_config(path: str, overrides=()) -> dict:
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        key, val = o.split("=", 1)
        *parents, leaf = key.split(".")
        d = cfg
        for p in parents:
            d = d[p]
        d[leaf] = yaml.safe_load(val)
    return cfg


def build_model(cfg: dict) -> QuantilePatchTST:
    d, m = cfg["data"], cfg["model"]
    return QuantilePatchTST(c_in=len(d["channels"]), seq_len=d["seq_len"], pred_len=d["pred_len"],
                            quantiles=d["quantiles"], target_ch=d["channels"].index(d["target"]), **m)


def make_loaders(cfg: dict, data: dict):
    d, t = cfg["data"], cfg["train"]
    bounds = SplitSpec(d["train_days"], d["val_days"]).bounds(data["x"].shape[1])
    tgt = d["channels"].index(d["target"])
    train = WindowDataset(data["x"], bounds["train"], d["seq_len"], d["pred_len"], d["train_stride"], tgt)
    val = WindowDataset(data["x"], bounds["val"], d["seq_len"], d["pred_len"], d["eval_stride"], tgt)
    n = min(len(train), t.get("windows_per_epoch") or len(train))
    train_dl = DataLoader(train, batch_size=t["batch_size"], num_workers=t.get("num_workers", 0),
                          sampler=RandomSampler(train, replacement=False, num_samples=n), drop_last=True)
    val_dl = DataLoader(val, batch_size=t["batch_size"] * 2, num_workers=t.get("num_workers", 0))
    return train_dl, val_dl, bounds


@torch.no_grad()
def evaluate_loss(model, dl, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        tot += pinball_loss(model(x), y, model.quantiles).item() * len(x)
        n += len(x)
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=3")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    t = cfg["train"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if t.get("num_threads"):
        torch.set_num_threads(t["num_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() and t.get("device", "auto") != "cpu" else "cpu")

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)

    data = load_npz(cfg["data"]["path"], cfg["data"]["channels"], cfg["data"].get("vm_limit", 0), cfg["seed"])
    train_dl, val_dl, bounds = make_loaders(cfg, data)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} vms={data['x'].shape[0]} train_windows/epoch={len(train_dl.sampler)} "
          f"val_windows={len(val_dl.dataset)} params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=t["epochs"] * len(train_dl))
    best, bad, history = float("inf"), 0, []
    for epoch in range(t["epochs"]):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y, _) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            loss = pinball_loss(model(x), y, model.quantiles)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            run += loss.item()
        val = evaluate_loss(model, val_dl, device)
        history.append({"epoch": epoch, "train_pinball": run / len(train_dl), "val_pinball": val,
                        "seconds": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if val < best:
            best, bad = val, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out / "best.pt")
        else:
            bad += 1
            if bad >= t.get("patience", 5):
                print("early stopping")
                break
    json.dump(history, open(out / "history.json", "w"), indent=1)
    print(f"best val pinball {best:.5f}; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
