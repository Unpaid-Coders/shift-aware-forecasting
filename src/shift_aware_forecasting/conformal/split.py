"""Offline split conformal baseline (split CQR, Romano et al., NeurIPS 2019).

Calibrates one fixed offset per interval level and horizon on the clean validation forecasts
(days 18-21) and never updates it. This is the standard non-adaptive correction that the online
methods are compared against: it fixes the average undercoverage of the base quantiles, but it
cannot react when the score distribution changes after a shift.

Two variants are written to one file:
  split     one offset per level and horizon, pooled over all VMs
  split_vm  one offset per level, horizon and VM, from that VM's own validation scores

For n calibration scores the offset is the ceil((n + 1)(1 - alpha))-th smallest score (the
largest score when that index exceeds n). Only observed targets are used.

  python -m shift_aware_forecasting.conformal.split --config configs/conformal_zaratan.yaml \\
      --predictions outputs/baseline_zaratan/predictions_val_s1.npz \\
      --out outputs/baseline_zaratan/split_offsets.npz

conformal.run --split-offsets <file> then evaluates both variants next to the online methods.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from .methods import cqr_scores
from .run import load


def conformal_quantile(scores: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    """Finite-sample split-conformal quantile along the last axis, using only `valid` entries.

    Rows without any valid score get NaN.
    """
    s = np.where(valid, scores, np.inf)
    s = np.sort(s, axis=-1)                                   # invalid entries sort to the end
    n = valid.sum(-1)
    k = np.ceil((n + 1) * (1 - alpha)).astype(np.int64)       # 1-based rank
    idx = np.clip(np.minimum(k, n) - 1, 0, None)
    q = np.take_along_axis(s, idx[..., None], axis=-1)[..., 0]
    return np.where(n > 0, q, np.nan)


def calibrate(pred_path: Path, levels: list) -> dict:
    d = load(pred_path, levels, None)
    V, H, T = d["y"].shape
    q_global = np.zeros((len(levels), H))
    q_vm = np.zeros((len(levels), V, H))
    for li, lv in enumerate(levels):
        s = cqr_scores(d["lo"][li], d["hi"][li], d["y"])      # [V, H, T]
        obs = d["obs"]
        for h in range(H):
            q_global[li, h] = conformal_quantile(s[:, h].ravel(), obs[:, h].ravel(), 1 - lv)
        q = conformal_quantile(s, obs, 1 - lv)                # [V, H]
        q_vm[li] = np.where(np.isfinite(q), q, q_global[li][None, :])  # VMs with no observed target
    return {"levels": np.array(levels), "q_global": q_global, "q_vm": q_vm, "vm_ids": d["vm_ids"],
            "n_steps": T}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path, help="clean validation forecasts (stride 1)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    res = calibrate(args.predictions, cfg["levels"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **res)
    for li, lv in enumerate(cfg["levels"]):
        qv = res["q_vm"][li]
        print(f"{lv:.0%}: pooled offset per horizon {np.round(res['q_global'][li], 3).tolist()}; "
              f"per-VM offset median {np.median(qv):.3f}, 90th pct {np.percentile(qv, 90):.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
