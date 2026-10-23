"""Exploratory analysis of the processed Azure 2019 CPU array.

Writes summary stats, figures and a table of candidate natural distribution shifts
(per-VM level / variance changes between consecutive multi-day windows).

  python scripts/explore_data.py --data data/azure2019_cpu_2000vm.npz --out docs/eda
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPD = 288  # steps per day


def acf(x: np.ndarray, lag: int) -> np.ndarray:
    a, b = x[:, :-lag], x[:, lag:]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).mean(1) / (a.std(1) * b.std(1) + 1e-8)


def shift_scan(avg: np.ndarray, win_days: int = 3) -> pd.DataFrame:
    """For each VM and each day boundary d, compare the win_days before and after d.
    Score = |mean change| / pooled std, and log std ratio. Returns the max per VM."""
    n, T = avg.shape
    days = T // SPD
    daily = avg[:, :days * SPD].reshape(n, days, SPD)
    rows = []
    for d in range(win_days, days - win_days + 1):
        before = daily[:, d - win_days:d].reshape(n, -1)
        after = daily[:, d:d + win_days].reshape(n, -1)
        pooled = np.sqrt(0.5 * (before.var(1) + after.var(1))) + 1e-3
        rows.append(pd.DataFrame({"vm": np.arange(n), "day": d,
                                  "mean_shift": (after.mean(1) - before.mean(1)) / pooled,
                                  "log_std_ratio": np.log((after.std(1) + 1e-3) / (before.std(1) + 1e-3))}))
    df = pd.concat(rows, ignore_index=True)  # unique index so idxmax picks one row per VM
    df["score"] = np.maximum(df.mean_shift.abs(), df.log_std_ratio.abs())
    return df.loc[df.groupby("vm").score.idxmax()].sort_values("score", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data)
    cpu, cat, obs = d["cpu"], d["category"], d["observed"]
    avg = cpu[..., 2]
    n, T, _ = cpu.shape
    summary = {"n_vms": int(n), "n_steps": int(T), "days": T / SPD,
               "observed_fraction": float(obs.mean()),
               "avg_cpu_mean": float(avg.mean()), "avg_cpu_median": float(np.median(avg)),
               "per_category": {}}
    ac1, ac_day, ac_week = acf(avg, 1), acf(avg, SPD), acf(avg, 7 * SPD)
    for c in np.unique(cat):
        m = cat == c
        summary["per_category"][str(c)] = {
            "n": int(m.sum()), "avg_cpu_mean": float(avg[m].mean()),
            "within_vm_std_mean": float(avg[m].std(1).mean()),
            "acf_lag1_median": float(np.median(ac1[m])),
            "acf_lag1day_median": float(np.median(ac_day[m])),
            "acf_lag1week_median": float(np.median(ac_week[m])),
            "frac_strong_daily_cycle(acf>0.5)": float((ac_day[m] > 0.5).mean())}

    shifts = shift_scan(avg)
    shifts["category"] = cat[shifts.vm.to_numpy()]
    shifts["vm_id"] = d["vm_ids"][shifts.vm.to_numpy()]
    shifts.to_csv(args.out / "shift_candidates.csv", index=False)
    test_start = 21
    strong = shifts[(shifts.score > 2) & (shifts.day >= test_start)]
    summary["vms_with_strong_shift_in_test_days"] = int(len(strong))
    summary["vms_with_strong_shift_anywhere"] = int((shifts.score > 2).sum())
    json.dump(summary, open(args.out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    days = np.arange(T) / SPD
    rng = np.random.default_rng(0)
    # example series per category
    fig, axes = plt.subplots(4, 2, figsize=(14, 9), sharex=True)
    for j, c in enumerate(["Interactive", "Delay-insensitive"]):
        for i, v in enumerate(rng.choice(np.where(cat == c)[0], 4, replace=False)):
            ax = axes[i, j]
            ax.fill_between(days, cpu[v, :, 0], cpu[v, :, 1], alpha=0.25, lw=0)
            ax.plot(days, avg[v], lw=0.5)
            ax.axvspan(18, 21, color="orange", alpha=0.08)
            ax.axvspan(21, days[-1], color="red", alpha=0.05)
            ax.set_title(f"{c} VM {v}", fontsize=8)
    axes[-1, 0].set_xlabel("day")
    axes[-1, 1].set_xlabel("day")
    fig.suptitle("Avg CPU % (line) and min-max band; shaded: val (orange) / test (red)")
    fig.tight_layout()
    fig.savefig(args.out / "example_series.png", dpi=110)

    # daily profile and aggregate load
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for c in np.unique(cat):
        m = cat == c
        prof = avg[m, :(T // SPD) * SPD].reshape(m.sum(), -1, SPD).mean((0, 1))
        ax[0].plot(np.arange(SPD) / 12, prof, label=c)
        ax[1].plot(days, avg[m].mean(0), lw=0.6, label=c)
    ax[0].set_xlabel("hour of day (trace time)")
    ax[0].set_ylabel("mean avg CPU %")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("day")
    ax[1].set_title("mean over VMs", fontsize=9)
    ax[2].hist(ac_day, bins=50)
    ax[2].set_xlabel("autocorrelation at 1-day lag (per VM)")
    fig.tight_layout()
    fig.savefig(args.out / "daily_profile_and_acf.png", dpi=110)

    # strongest natural shifts
    top = shifts.head(6)
    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex=True)
    for ax, (_, r) in zip(axes.flat, top.iterrows()):
        ax.plot(days, avg[int(r.vm)], lw=0.5)
        ax.axvline(r.day, color="r", ls="--")
        ax.set_title(f"VM {int(r.vm)} ({r.category}) mean_shift={r.mean_shift:.1f} "
                     f"log_std_ratio={r.log_std_ratio:.2f}", fontsize=8)
    fig.suptitle("Strongest candidate natural shifts (3-day before/after windows)")
    fig.tight_layout()
    fig.savefig(args.out / "shift_candidates.png", dpi=110)


if __name__ == "__main__":
    main()
