"""Figures and a markdown summary from the outputs of shift_aware_forecasting.conformal.run.

  python -m shift_aware_forecasting.conformal.plots --out outputs/conformal_zaratan
"""
import argparse
import json
from pathlib import Path

import numpy as np

STEPS_PER_DAY = 288  # 5-minute readings (same as data/windows.py; not imported to avoid loading torch)
from .methods import METHOD_LABELS

# Fixed categorical order (base model in neutral gray).
COLORS = {"base": "#8a8984", "aci": "#2a78d6", "pid": "#eb6834", "qt": "#1baf7a", "trail": "#eda100"}
ORDER = ["base", "trail", "aci", "qt", "pid"]


def _style(ax):
    ax.grid(True, color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#a9a8a2")
    ax.tick_params(colors="#52514e", labelsize=8)


def smooth(v, win):
    """Centered moving mean that ignores NaN."""
    ok = np.isfinite(v)
    k = np.ones(win)
    num = np.convolve(np.where(ok, v, 0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def aligned(values, obs, rel, lo, hi):
    """Average per-VM values [V, T] over VMs at each step relative to the shift, in [lo, hi)."""
    R = np.arange(lo, hi)
    acc = np.zeros(len(R)); cnt = np.zeros(len(R))
    for v in range(values.shape[0]):
        j = rel[v] - lo
        m = (j >= 0) & (j < len(R)) & obs[v]
        np.add.at(acc, j[m], values[v][m]); np.add.at(cnt, j[m], 1)
    with np.errstate(invalid="ignore"):
        return R, np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), cnt


def fig_shift_aligned(tr, sel, level, h, path, before_h=24, after_h=72):
    import matplotlib.pyplot as plt
    ss = tr["shift_step"]
    pre = ss >= int(tr["pre_min_step"])      # VMs with >= 1 evaluated day before the shift
    rel = (tr["ts"][None, :] + h - 1) - ss[:, None]
    obs = tr[f"shift_obs__h{h}"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    lo, hi = -before_h * 12, after_h * 12
    for meth in ORDER:
        if meth not in sel:
            continue
        n = sel[meth]
        cov = tr[f"shift_cov__{n}__h{h}"].astype(float)
        wid = tr[f"shift_wid__{n}__h{h}"].astype(float)
        R, c, cnt = aligned(cov[pre], obs[pre], rel[pre], lo, hi)
        _, w, _ = aligned(wid[pre], obs[pre], rel[pre], lo, hi)
        hours = R / 12
        lw = 1.2 if meth == "base" else 1.6
        ax[0].plot(hours, smooth(c, 12), color=COLORS[meth], lw=lw, label=METHOD_LABELS[meth])
        ax[1].plot(hours, smooth(w, 12), color=COLORS[meth], lw=lw)
    ax[0].axhline(level, color="#0b0b0b", ls="--", lw=0.9)
    for a in ax:
        a.axvline(0, color="#52514e", lw=0.9, ls=":")
        _style(a)
    ax[0].set_ylabel("coverage (1-hour mean)", fontsize=9, color="#52514e")
    ax[1].set_ylabel("interval width (CPU %)", fontsize=9, color="#52514e")
    ax[1].set_xlabel("hours since the shift", fontsize=9, color="#52514e")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=5,
                 handlelength=1.5, columnspacing=1.2)
    fig.suptitle(f"{level:.0%} interval (dashed: target), {h}-step ahead, {int(pre.sum())} VMs with a natural shift, "
                 f"aligned at the shift", fontsize=10, x=0.02, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_over_time(tr, sel, level, path, h=1):
    import matplotlib.pyplot as plt
    days = (tr["ts"] + h - 1) / STEPS_PER_DAY
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for meth in ORDER:
        if meth not in sel:
            continue
        n = sel[meth]
        ax[0].plot(days, smooth(tr[f"time_cov__{n}"][h - 1], 12), color=COLORS[meth], lw=1.1, label=METHOD_LABELS[meth])
        ax[1].plot(days, smooth(tr[f"time_wid__{n}"][h - 1], 12), color=COLORS[meth], lw=1.1)
    ax[0].axhline(level, color="#0b0b0b", ls="--", lw=0.9)
    for a in ax:
        _style(a)
    ax[0].set_ylabel(f"coverage, h={h} (1-hour mean)", fontsize=9, color="#52514e")
    ax[1].set_ylabel("interval width (CPU %)", fontsize=9, color="#52514e")
    ax[1].set_xlabel("day of trace", fontsize=9, color="#52514e")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=5)
    fig.suptitle(f"{level:.0%} interval (dashed: target), all VMs, test days", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_by_horizon(metrics, sel, level, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for a, g, title in zip(ax, ("tune", "shift"), ("VMs without a detected shift", "VMs with a natural shift")):
        for meth in ORDER:
            if meth not in sel:
                continue
            c = metrics[sel[meth]][g]["per_horizon_coverage"]
            a.plot(np.arange(1, len(c) + 1), c, color=COLORS[meth], lw=1.6, marker="o", ms=4, label=METHOD_LABELS[meth])
        a.axhline(level, color="#0b0b0b", ls="--", lw=0.9)
        a.set_title(title, fontsize=10, loc="left")
        a.set_xlabel("horizon (5-min steps)", fontsize=9, color="#52514e")
        _style(a)
    ax[0].set_ylabel("coverage", fontsize=9, color="#52514e")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_md(res) -> str:
    meta, out = res["meta"], []
    out.append(f"# Online conformal results\n\n{meta['n_vms']} VMs x {meta['n_steps']} test steps x "
               f"{meta['horizons']} horizons. Shift VMs: {meta['n_shift_vms']} "
               f"({meta['n_shift_vms_with_1d_before']} with at least 1 day of test data before the shift; "
               f"before/after numbers use those). Hyperparameters were selected on the non-shift VMs only.\n")
    for L, sel in res["selected"].items():
        out.append(f"\n## {L} interval\n")
        out.append("| method | config | coverage (all) | width (all) | local cov. error | coverage (shift VMs) "
                   "| 1 day before shift | 1 day after shift | after shift (all) | width after shift |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for meth in ORDER:
            if meth not in sel:
                continue
            m = res["metrics"][L][sel[meth]]
            sw = m["shift_windows"]
            out.append(f"| {METHOD_LABELS[meth]} | `{sel[meth]}` | {m['all']['coverage']:.3f} | {m['all']['width']:.2f} "
                       f"| {m['all']['local_coverage_error']:.3f} | {m['shift']['coverage']:.3f} "
                       f"| {sw['before_1d']['coverage']:.3f} | {sw['after_1d']['coverage']:.3f} "
                       f"| {sw['after_all']['coverage']:.3f} | {sw['after_all']['width']:.2f} |")
    out.append("\nLocal coverage error = mean over VMs and days of |daily coverage - target|. "
               "All numbers average over horizons 1-12 unless noted. Full sweep: table_*.csv.\n")
    return "\n".join(out)


def make_all(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    res = json.load(open(out / "metrics.json"))
    for L, sel in res["selected"].items():
        level = int(L.rstrip("%")) / 100
        tag = L.rstrip("%")
        tr = dict(np.load(out / f"traces_{tag}.npz"))
        for h in tr["plot_horizons"]:
            if f"shift_obs__h{h}" in tr and len(tr["shift_step"]):
                fig_shift_aligned(tr, sel, level, int(h), out / f"fig_shift_aligned_{tag}_h{h}.png")
        fig_over_time(tr, sel, level, out / f"fig_coverage_over_time_{tag}_h1.png")
        fig_by_horizon(res["metrics"][L], sel, level, out / f"fig_coverage_by_horizon_{tag}.png")
    (out / "results.md").write_text(summary_md(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    make_all(ap.parse_args().out)
