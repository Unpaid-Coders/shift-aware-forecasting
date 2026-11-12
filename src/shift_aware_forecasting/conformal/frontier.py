"""Width-vs-error frontier over the whole conformal hyperparameter sweep.

Each point is one setting of one method (base, trail, ACI, QT, PID), read from the table_<level>.csv
and shift_table_<level>.csv files that shift_aware_forecasting.conformal.run writes. One row of
panels per run:
  left   local coverage error (mean |daily per-VM coverage - target|) against average width
  right  coverage in the day after the shift against average width in that day (dashed: target)
Settings that are better sit lower-left (left panel) or higher-left (right panel). Points of a
method are joined in order of width (PID points are not joined: its step size x gain grid has no order). ACI and QT points are labelled with their step size; PID points
are left unlabelled because at equal step size they sit on top of QT's. The base quantiles are far
outside the plotted range, so their values are written in the panel corner instead of plotted.

  python -m shift_aware_forecasting.conformal.frontier --levels 90 80 --out outputs/frontier \\
      --runs outputs/conformal_zaratan="clean data (70 VMs with a natural shift)" \\
             outputs/conformal_shift_scale="injected variance shift" \\
             outputs/conformal_shift_level_scale="injected level and variance shift"
"""
import argparse
from pathlib import Path

import pandas as pd

from .methods import METHOD_LABELS
from .plots import COLORS, ORDER, _style


def load(run: Path, level: str) -> pd.DataFrame:
    t = pd.read_csv(run / f"table_{level}.csv")
    t = t[t.group == "all"].set_index("config")
    s = pd.read_csv(run / f"shift_table_{level}.csv")
    s = s.pivot(index="config", columns="window", values=["coverage", "width"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    return t.join(s)


def short(config: str, method: str) -> str:
    """aci_lr0.005 -> 0.005, pid_lr0.01_ki0.2 -> 0.01/0.2, base/trail -> ''."""
    rest = config[len(method):].lstrip("_")
    return rest.replace("lr", "").replace("_ki", "/")


def fig_frontier(runs: list, level: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = int(level) / 100
    fig, ax = plt.subplots(len(runs), 2, figsize=(10, 3.6 * len(runs)), squeeze=False)
    for r, (run, label) in enumerate(runs):
        d = load(run, level)
        panels = [(ax[r, 0], "width", "local_coverage_error", "average width (CPU %)", "local coverage error"),
                  (ax[r, 1], "width_after_1d", "coverage_after_1d", "average width, day after the shift (CPU %)",
                   "coverage, day after the shift")]
        for a, x, y, xl, yl in panels:
            for meth in ORDER:
                m = d[d.method == meth].sort_values(x)
                if m.empty:
                    continue
                if meth == "base":
                    b = m.iloc[0]
                    a.text(0.98, 0.04 if y == "coverage_after_1d" else 0.96,
                           f"base quantiles: {b[y]:.3f} at width {b[x]:.1f}", transform=a.transAxes,
                           ha="right", va="bottom" if y == "coverage_after_1d" else "top", fontsize=7.5,
                           color=COLORS["base"])
                    continue
                # PID's grid (step size x integrator gain) has no natural order, so no joining line
                a.plot(m[x], m[y], color=COLORS[meth], lw=0 if meth == "pid" else 1.4, marker="o", ms=4.5,
                       label=METHOD_LABELS[meth] if (r, a) == (0, ax[0, 0]) else None)
                if meth in ("aci", "qt"):
                    for cfg, row in m.iterrows():
                        a.annotate(short(cfg, meth), (row[x], row[y]), xytext=(4, -9 if meth == "qt" else 4),
                                   textcoords="offset points", fontsize=6.5, color=COLORS[meth])
            if y == "coverage_after_1d":
                a.axhline(target, color="#0b0b0b", ls="--", lw=0.9)
            a.set_xlabel(xl, fontsize=9, color="#52514e")
            a.set_ylabel(yl, fontsize=9, color="#52514e")
            _style(a)
        ax[r, 0].set_title(label, fontsize=10, loc="left")
    fig.suptitle(f"{target:.0%} interval: every setting in the sweep (labels: step size)",
                 fontsize=10, x=0.02, ha="left", color="#0b0b0b")
    fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04 * 3 / len(runs), 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="output dirs of conformal.run, each as DIR or DIR=LABEL")
    ap.add_argument("--levels", nargs="+", default=["90", "80"], help="interval levels in percent")
    ap.add_argument("--out", type=Path, default=Path("outputs/frontier"))
    args = ap.parse_args()
    runs = []
    for spec in args.runs:
        d, _, label = spec.partition("=")
        runs.append((Path(d), label or Path(d).name))
    args.out.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        path = args.out / f"frontier_{level}.png"
        fig_frontier(runs, level, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
