"""Paper figures for the online conformal experiments, from the outputs of conformal.run.

  fig_recovery_<L>_h<h>   coverage and width aligned at an injected variance shift (horizon h), with
                          the settings selected on clean data (left) and on a validation shift (right)
  fig_frontier_<L>        every setting in the sweep, with both selections marked: local coverage
                          error on clean data (left) and coverage in the day after a variance shift (right)
  fig_shift_size_<L>      coverage in the day after the shift and overall width, for clean data and
                          1.5x / 3x variance shifts, under both selections
  results_<L>.csv         the numbers behind fig_shift_size

Each figure is written as PDF (for the paper) and PNG. Missing runs are skipped with a message.

  python -m shift_aware_forecasting.conformal.paper_figures --out outputs/paper_figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .frontier import load as load_sweep
from .methods import METHOD_LABELS
from .plots import COLORS, _style, aligned, smooth

INK, INK2, GRID = "#0b0b0b", "#52514e", "#a9a8a2"
METHODS = ["base", "aci", "qt", "pid"]  # trail is dominated in every comparison, so it is left out
LINESTYLE = {"base": "-", "aci": "-", "qt": "-", "pid": (0, (4, 2))}  # PID sits on QT: dash it

# (name, directory, test condition, selection)
RUNS = [
    ("clean_orig", "outputs/conformal_zaratan", "clean", "clean data"),
    ("clean_val", "outputs/conformal_zaratan_valsel", "clean", "validation shift"),
    ("x15_orig", "outputs/conformal_shift_scale_x1.5", "1.5x variance shift", "clean data"),
    ("x15_val", "outputs/conformal_shift_scale_x1.5_valsel", "1.5x variance shift", "validation shift"),
    ("x3_orig", "outputs/conformal_shift_scale", "3x variance shift", "clean data"),
    ("x3_val", "outputs/conformal_shift_scale_valsel", "3x variance shift", "validation shift"),
]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def _save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", out / f"{name}.pdf")


def fig_recovery(root: Path, out: Path, L: str, h: int, before_h=12, after_h=48, smooth_h=1) -> None:
    plt = _plt()
    tag = L.rstrip("%")
    level = int(tag) / 100
    panels = [("Settings selected on clean data", root / RUNS[4][1]),
              ("Settings selected on a validation shift", root / RUNS[5][1])]
    if not all((d / f"traces_{tag}.npz").exists() for _, d in panels):
        print("skip fig_recovery: missing", [str(d) for _, d in panels]); return
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey="row",
                           gridspec_kw={"height_ratios": [3, 2]})
    lo, hi = -before_h * 12, after_h * 12
    for col, (title, d) in enumerate(panels):
        tr = dict(np.load(d / f"traces_{tag}.npz"))
        sel = json.load(open(d / "selected.json"))[L]
        ss = tr["shift_step"]
        pre = ss >= int(tr["pre_min_step"])
        rel = (tr["ts"][None, :] + h - 1) - ss[:, None]
        obs = tr[f"shift_obs__h{h}"]
        for meth in METHODS:
            n = sel[meth]
            cov = tr[f"shift_cov__{n}__h{h}"].astype(float)
            wid = tr[f"shift_wid__{n}__h{h}"].astype(float)
            R, c, _ = aligned(cov[pre], obs[pre], rel[pre], lo, hi)
            _, w, _ = aligned(wid[pre], obs[pre], rel[pre], lo, hi)
            kw = dict(color=COLORS[meth], lw=1.1 if meth == "base" else 1.5, ls=LINESTYLE[meth])
            label = METHOD_LABELS[meth] + ("" if meth == "base" else f" ({short(n, meth)})")
            ax[0, col].plot(R / 12, smooth(c, 12 * smooth_h), **kw)
            ax[1, col].plot(R / 12, smooth(w, 12 * smooth_h), label=label, **kw)
        ax[0, col].axhline(level, color=INK, ls="--", lw=0.8)
        ax[0, col].set_title(title, loc="left", color=INK)
        ax[1, col].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=6.5,
                          frameon=False, handlelength=1.8, columnspacing=1.0)
        ax[1, col].set_xlabel("hours since the shift", color=INK2)
        for a in ax[:, col]:
            a.axvline(0, color=INK2, lw=0.8, ls=":")
            _style(a)
    ax[0, 0].set_ylabel(f"coverage ({smooth_h} h mean)", color=INK2)
    ax[1, 0].set_ylabel("width (CPU %)", color=INK2)
    fig.tight_layout()
    _save(fig, out, f"fig_recovery_{tag}_h{h}")
    plt.close(fig)


def short(config: str, method: str) -> str:
    """aci_lr0.005 -> lr 0.005, pid_lr0.01_ki0.2 -> lr 0.01, ki 0.2."""
    rest = config[len(method):].lstrip("_")
    return rest.replace("lr", "lr ").replace("_ki", ", ki ")


def fig_frontier(root: Path, out: Path, L: str) -> None:
    plt = _plt()
    tag = L.rstrip("%")
    need = [root / RUNS[0][1], root / RUNS[4][1], root / RUNS[1][1]]
    if not all((d / f"table_{tag}.csv").exists() for d in need[:2]) or not (need[2] / "selected.json").exists():
        print("skip fig_frontier: missing", [str(d) for d in need]); return
    sel_clean = json.load(open(root / RUNS[0][1] / "selected.json"))[L]
    sel_val = json.load(open(root / RUNS[1][1] / "selected.json"))[L]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    panels = [(ax[0], root / RUNS[0][1], "width", "local_coverage_error", "average width (CPU %)",
               "local coverage error", "Clean test data"),
              (ax[1], root / RUNS[4][1], "width_after_1d", "coverage_after_1d",
               "average width, day after the shift (CPU %)", "coverage, day after the shift",
               "3x variance shift")]
    for a, d, x, y, xl, yl, title in panels:
        t = load_sweep(d, tag)
        for meth in ["aci", "qt", "pid"]:
            m = t[t.method == meth].sort_values(x)
            a.plot(m[x], m[y], color=COLORS[meth], lw=0 if meth == "pid" else 1.0, marker="o", ms=3.5,
                   alpha=0.85, label=METHOD_LABELS[meth] if a is ax[0] else None)
            for sel, mk in ((sel_clean, "s"), (sel_val, "*")):
                r = t.loc[sel[meth]]
                big = {"aci": 1.0, "qt": 1.25, "pid": 0.75}[meth]  # QT and PID often coincide: nest them
                a.plot(r[x], r[y], marker=mk, ms=(7 if mk == "s" else 11) * big,
                       mfc="none" if mk == "s" else COLORS[meth],
                       mec=INK if mk == "*" else COLORS[meth], mew=1.0 if mk == "*" else 1.4, ls="none",
                       zorder={"qt": 5, "aci": 6, "pid": 7}[meth])
        b = t.loc["base"]
        after = y == "coverage_after_1d"
        a.text(0.98, 0.04 if after else 0.96, f"base: {b[y]:.3f} at width {b[x]:.1f}",
               transform=a.transAxes, ha="right", va="bottom" if after else "top",
               fontsize=6.5, color=INK2)
        if after:
            a.axhline(int(tag) / 100, color=INK, ls="--", lw=0.8)
        a.set_xlabel(xl, color=INK2); a.set_ylabel(yl, color=INK2)
        a.set_title(title, loc="left", color=INK)
        _style(a)
    a = ax[0]
    a.plot([], [], marker="s", mfc="none", mec=INK2, ls="none", ms=6, label="selected on clean data")
    a.plot([], [], marker="*", mfc=INK2, mec=INK, ls="none", ms=9, label="selected on validation shift")
    fig.legend(loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, f"fig_frontier_{tag}")
    plt.close(fig)


def results_table(root: Path, L: str) -> pd.DataFrame:
    rows = []
    for name, d, cond, selection in RUNS:
        p = root / d / "metrics.json"
        if not p.exists():
            print("skip", d, "(no metrics.json)"); continue
        res = json.load(open(p))
        for meth in METHODS:
            n = res["selected"][L][meth]
            m = res["metrics"][L][n]
            rows.append({"condition": cond, "selection": selection, "method": meth, "config": n,
                         "coverage": m["all"]["coverage"], "width": m["all"]["width"],
                         "local_err": m["all"]["local_coverage_error"],
                         "coverage_after_1d": m["shift_windows"]["after_1d"]["coverage"],
                         "width_after_1d": m["shift_windows"]["after_1d"]["width"]})
    return pd.DataFrame(rows)


def fig_shift_size(df: pd.DataFrame, out: Path, L: str) -> None:
    plt = _plt()
    tag = L.rstrip("%")
    if df.empty:
        print("skip fig_shift_size: no runs"); return
    conds = [c for c in ["clean", "1.5x variance shift", "3x variance shift"] if c in set(df.condition)]
    shifted = [c for c in conds if c != "clean"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2, 3]})
    off = {m: (i - 1.5) * 0.17 for i, m in enumerate(METHODS)}
    for a, y, conds_a, yl in ((ax[0], "coverage_after_1d", shifted, "coverage, day after the shift"),
                              (ax[1], "width", conds, "average width (CPU %)")):
        for meth in METHODS:
            for selection, filled in (("clean data", False), ("validation shift", True)):
                if meth == "base" and not filled:
                    continue  # base has no settings: one point per condition
                s = df[(df.method == meth) & (df.selection == selection)].set_index("condition")
                xs = [i + off[meth] for i, c in enumerate(conds_a) if c in s.index]
                ys = [s.loc[c, y] for c in conds_a if c in s.index]
                a.plot(xs, ys, ls="none", marker="o", ms=5.5, mew=1.3, color=COLORS[meth],
                       mfc=COLORS[meth] if filled else "white",
                       label=(METHOD_LABELS[meth] if filled else None) if a is ax[1] else None)
        a.set_xticks(range(len(conds_a)), [c.replace(" variance shift", " shift") for c in conds_a])
        a.set_xlim(-0.6, len(conds_a) - 0.4)
        a.set_ylabel(yl, color=INK2)
        _style(a)
        a.grid(False, axis="x")
    ax[0].set_title("Recovery after the shift", loc="left", color=INK)
    ax[1].set_title("Cost in width", loc="left", color=INK)
    ax[0].axhline(int(tag) / 100, color=INK, ls="--", lw=0.8)
    ax[1].plot([], [], ls="none", marker="o", mfc="white", mec=INK2, mew=1.3, label="selected on clean data")
    ax[1].plot([], [], ls="none", marker="o", color=INK2, label="selected on validation shift")
    fig.legend(loc="lower center", ncol=6, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, f"fig_shift_size_{tag}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root that holds outputs/")
    ap.add_argument("--out", type=Path, default=Path("outputs/paper_figures"))
    ap.add_argument("--levels", nargs="+", default=["90", "80"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for tag in args.levels:
        L = f"{tag}%"
        for h in args.horizons:
            fig_recovery(args.root, args.out, L, h)
        fig_frontier(args.root, args.out, L)
        df = results_table(args.root, L)
        df.to_csv(args.out / f"results_{tag}.csv", index=False, float_format="%.4f")
        fig_shift_size(df, args.out, L)


if __name__ == "__main__":
    main()
