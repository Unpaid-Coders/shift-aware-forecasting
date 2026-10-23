"""Evaluate a trained checkpoint on the test split and save rolling forecasts.

Saves predictions.npz (pred [N, H, Q], y [N, H], vm [N], t [N] = first target step,
all in CPU %) so the online conformal wrapper can replay them in time order, plus
metrics.json and a coverage-over-time plot.

  python -m msml612.evaluate --run outputs/baseline_cpu_small [--stride 1] [--vm-limit 50]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.windows import STEPS_PER_DAY, SplitSpec, WindowDataset, load_npz
from .metrics import interval_metrics, point_metrics
from .train import build_model


@torch.no_grad()
def predict(model, ds, batch_size, device):
    preds, ys, idx = [], [], []
    model.eval()
    for x, y, i in DataLoader(ds, batch_size=batch_size):
        preds.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
        idx.append(i.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(idx)


def naive_baselines(x: np.ndarray, vm: np.ndarray, t: np.ndarray, H: int, tgt: int) -> dict:
    steps = np.arange(H)
    last = x[vm, t - 1, tgt][:, None].repeat(H, 1)                        # persistence
    seasonal = x[vm[:, None], (t[:, None] + steps - STEPS_PER_DAY), tgt]  # same time yesterday
    return {"persistence": last, "seasonal_naive_1d": seasonal}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--vm-limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    ck = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    d = cfg["data"]
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vm_limit = args.vm_limit if args.vm_limit is not None else d.get("vm_limit", 0)
    data = load_npz(d["path"], d["channels"], vm_limit, cfg["seed"])
    x = data["x"]
    bounds = SplitSpec(d["train_days"], d["val_days"]).bounds(x.shape[1])
    tgt = d["channels"].index(d["target"])
    stride = args.stride or d["eval_stride"]
    ds = WindowDataset(x, bounds[args.split], d["seq_len"], d["pred_len"], stride, tgt)
    pred, y, idx = predict(model, ds, args.batch_size, device)
    vm, t = idx[:, 0], idx[:, 1]
    pred, y = pred * 100, y * 100  # back to CPU %

    q = [round(float(v), 4) for v in model.quantiles.cpu()]  # float32 -> exact 0.05, 0.95
    obs = data["observed"][vm[:, None], t[:, None] + np.arange(d["pred_len"])]  # skip gap-filled targets
    metrics = {"n_windows": int(len(y)), "n_vms": int(x.shape[0]), "split": args.split, "stride": stride,
               "model_median": point_metrics(pred[..., q.index(0.5)], y, obs)}
    for name, yhat in naive_baselines(x, vm, t, d["pred_len"], tgt).items():
        metrics[name] = point_metrics(yhat * 100, y, obs)
    for lo_q in [ql for ql in q if ql < 0.5]:
        hi_q = round(1 - lo_q, 4)
        if hi_q in q:
            level = f"{round(100 * (hi_q - lo_q))}%"
            lo, hi = pred[..., q.index(lo_q)], pred[..., q.index(hi_q)]
            m = interval_metrics(lo, hi, y, obs)
            m["per_horizon_coverage"] = [interval_metrics(lo[:, h], hi[:, h], y[:, h], obs[:, h])["coverage"]
                                         for h in range(d["pred_len"])]
            metrics[f"interval_{level}"] = m
    print(json.dumps({k: v for k, v in metrics.items()}, indent=1))

    tag = f"{args.split}_s{stride}"
    np.savez_compressed(args.run / f"predictions_{tag}.npz", pred=pred.astype(np.float32),
                        y=y.astype(np.float32), vm=vm, t=t, observed=obs, quantiles=np.array(q),
                        vm_ids=data["vm_ids"], category=data["category"])
    json.dump(metrics, open(args.run / f"metrics_{tag}.json", "w"), indent=1)
    plot_coverage(pred, y, vm, t, q, args.run / f"coverage_over_time_{tag}.png", stride)


def plot_coverage(pred, y, vm, t, q, path, stride, level=(0.05, 0.95)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if level[0] not in q or level[1] not in q:
        return
    cov = (y[:, 0] >= pred[:, 0, q.index(level[0])]) & (y[:, 0] <= pred[:, 0, q.index(level[1])])
    width = pred[:, 0, q.index(level[1])] - pred[:, 0, q.index(level[0])]
    ts = np.unique(t)
    pos = np.searchsorted(ts, t)
    cov_t = np.bincount(pos, cov, len(ts)) / np.bincount(pos, minlength=len(ts))
    w_t = np.bincount(pos, width, len(ts)) / np.bincount(pos, minlength=len(ts))
    win = max(1, 12 // stride)  # ~1 hour smoothing
    days = ts / STEPS_PER_DAY
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(days, cov_t, alpha=0.3, lw=0.7, label="per step (avg over VMs)")
    ax[0].plot(days[win - 1:], rolling_coverage_mean(cov_t, win), lw=1.5, label=f"{win}-step mean")
    ax[0].axhline(level[1] - level[0], color="k", ls="--", lw=1, label="target")
    ax[0].set_ylabel("coverage (h=1)")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[1].plot(days, w_t, lw=0.8)
    ax[1].set_ylabel("interval width (CPU %)")
    ax[1].set_xlabel("day of trace")
    fig.suptitle(f"Baseline {round(100 * (level[1] - level[0]))}% interval, 1-step ahead, test split")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def rolling_coverage_mean(v, win):
    c = np.cumsum(np.insert(v, 0, 0))
    return (c[win:] - c[:-win]) / win


if __name__ == "__main__":
    main()
