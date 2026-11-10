"""Wrap the baseline's rolling test forecasts in online conformal calibration and evaluate,
with a focus on the VMs that have a natural distribution shift in the test days.

Input: predictions_<split>_s1.npz from shift_aware_forecasting.evaluate (stride 1, so every VM
has one forecast per step). For each interval level, horizon h and VM, the CQR scores of the base
interval form one time-ordered sequence; each method turns it into offsets q (see methods.py) using only
scores observed when the forecast is made (ahead = h).

VMs are split into
  shift  VMs with a strong level/variance shift in the test days (shift_candidates.csv,
         score > shift_min_score, day >= 21); held out, used for the shift tables and figures
  tune   all other VMs; used to pick one hyperparameter setting per method
Outputs in out_dir: metrics.json, selected.json, table_<level>.csv (all configs, all groups),
shift_table_<level>.csv (before/after shift), traces_<level>.npz, and figures (plots.py).

  python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml [--vm-limit 200]
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STEPS_PER_DAY = 288  # 5-minute readings (same as data/windows.py; not imported to avoid loading torch)
from .methods import MethodConfig, cqr_scores, online_offsets

TEST_START_DAY = 21
G = {}  # data shared with forked workers


def level_name(level: float) -> str:
    return f"{round(100 * level)}%"


def load(path: Path, levels: list, vm_limit: int | None) -> dict:
    d = np.load(path, allow_pickle=False)
    vm, t = d["vm"], d["t"]
    vms, ts = np.unique(vm), np.unique(t)
    if vm_limit:
        vms = vms[:vm_limit]
    if not np.all(np.diff(ts) == 1):
        raise ValueError("forecast origins are not contiguous; run shift_aware_forecasting.evaluate with --stride 1")
    V, T = len(vms), len(ts)
    keep = np.isin(vm, vms)
    vi, ti = np.searchsorted(vms, vm[keep]), np.searchsorted(ts, t[keep])
    if keep.sum() != V * T:
        raise ValueError(f"expected {V}x{T} forecasts, found {keep.sum()}")
    order = np.lexsort((ti, vi))  # VM-major, then time
    pred = d["pred"][keep][order]
    H = pred.shape[1]
    pred = pred.reshape(V, T, H, -1).transpose(0, 2, 1, 3)     # [V, H, T, Q]
    y = d["y"][keep][order].reshape(V, T, H).transpose(0, 2, 1)
    obs = d["observed"][keep][order].reshape(V, T, H).transpose(0, 2, 1)
    qs = [round(float(x), 4) for x in d["quantiles"]]
    lo, hi = [], []
    for lv in levels:
        a, b = round((1 - lv) / 2, 4), round((1 + lv) / 2, 4)
        if a not in qs or b not in qs:
            raise ValueError(f"level {lv} needs quantiles {a} and {b}, model has {qs}")
        lo.append(np.ascontiguousarray(pred[..., qs.index(a)]))
        hi.append(np.ascontiguousarray(pred[..., qs.index(b)]))
    return {"lo": lo, "hi": hi, "y": np.ascontiguousarray(y), "obs": np.ascontiguousarray(obs),
            "vms": vms, "ts": ts, "vm_ids": d["vm_ids"][vms], "category": d["category"][vms]}


def shift_info(csv: Path, vm_ids: np.ndarray, min_score: float) -> np.ndarray:
    """Shift time step per VM (-1 = no strong shift in the test days). Matched by VM id."""
    df = pd.read_csv(csv)
    df = df[(df.score > min_score) & (df.day >= TEST_START_DAY)]
    step = dict(zip(df.vm_id, df.day * STEPS_PER_DAY))
    return np.array([step.get(v, -1) for v in vm_ids], dtype=np.int64)


def build_configs(cfg: dict) -> list:
    return [MethodConfig(method=m, **kw) for m, lst in cfg["methods"].items() for kw in (lst or [{}])]


def work(task):
    li, h = task
    lv, cfg = G["levels"][li], G["cfg"]
    lo, hi = G["lo"][li][:, h], G["hi"][li][:, h]
    y, obs = G["y"][:, h], G["obs"][:, h]
    V, T = y.shape
    tt = G["ts"] + h                                # target time of each forecast
    obs = obs & (tt >= G["eval_start"])[None, :]    # metrics skip the burn-in
    s = cqr_scores(lo, hi, y)
    t0 = time.time()
    qs = online_offsets(s, 1 - lv, h + 1, G["configs"], cfg["window"], lower=-(hi - lo) / 2)
    qs = {"base": np.zeros_like(s), **qs}
    day = tt // STEPS_PER_DAY
    days = np.unique(day[tt >= G["eval_start"]])
    shift_step = G["shift_step"]
    groups = {"all": np.ones(V, bool), "tune": shift_step < 0, "shift": shift_step >= 0}
    rel = tt[None, :] - shift_step[:, None]        # steps since the shift (shift VMs only)
    pre = shift_step >= G["eval_start"] + STEPS_PER_DAY   # >= 1 evaluated day before the shift
    win = {"before_1d": (rel >= -STEPS_PER_DAY) & (rel < 0),
           "after_1d": (rel >= 0) & (rel < STEPS_PER_DAY),
           "after_all": rel >= 0}
    save_h = (h + 1) in cfg["plot_horizons"]
    res = {"stats": {}, "time": {}, "shift": {}, "trace": {}}
    for name, q in qs.items():
        cov = s <= q
        wid = np.maximum(hi - lo + 2 * q, 0)
        st = {}
        for g, gm in groups.items():
            m = obs & gm[:, None]
            # per-VM daily coverage -> local coverage error
            with np.errstate(invalid="ignore", divide="ignore"):
                dcov = np.stack([(cov & m)[:, day == d].sum(1) / m[:, day == d].sum(1) for d in days], 1)[gm]
            ok = np.isfinite(dcov)
            st[g] = {"cov": float(cov[m].sum()), "wid": float(wid[m].sum()), "n": int(m.sum()),
                     "local_err": float(np.abs(dcov[ok] - lv).mean()) if ok.any() else np.nan,
                     "vmday_under10": float((dcov[ok] < lv - 0.1).mean()) if ok.any() else np.nan}
        res["stats"][name] = st
        m = obs
        res["time"][name] = ((cov & m).sum(0) / np.maximum(m.sum(0), 1), (wid * m).sum(0) / np.maximum(m.sum(0), 1))
        sw = {}
        for w, wm in win.items():
            mm = obs & wm & pre[:, None]
            sw[w] = {"cov": float(cov[mm].sum()), "wid": float(wid[mm].sum()), "n": int(mm.sum())}
        res["shift"][name] = sw
        if save_h:
            sv = groups["shift"]
            res["trace"][name] = (cov[sv], wid[sv].astype(np.float16), obs[sv])
    print(f"level {level_name(lv)} h={h + 1}: {time.time() - t0:.0f}s", flush=True)
    return task, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--predictions", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--vm-limit", type=int, default=None, help="first N VMs only (quick test)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pred_path = args.predictions or Path(cfg["predictions"])
    out = args.out or Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    levels = cfg["levels"]
    t0 = time.time()
    data = load(pred_path, levels, args.vm_limit)
    shift_step = shift_info(Path(cfg["shift_csv"]), data["vm_ids"], cfg["shift_min_score"])
    V, H, T = data["y"].shape
    print(f"loaded {V} VMs x {H} horizons x {T} steps in {time.time() - t0:.0f}s; "
          f"{(shift_step >= 0).sum()} shift VMs", flush=True)
    configs = build_configs(cfg)
    eval_start = int((TEST_START_DAY + cfg.get("burnin_days", 0)) * STEPS_PER_DAY)
    pre_min = eval_start + STEPS_PER_DAY
    G.update(data, levels=levels, cfg=cfg, configs=configs, shift_step=shift_step, eval_start=eval_start)
    tasks = [(li, h) for li in range(len(levels)) for h in range(H)]
    workers = args.workers or cfg.get("workers", 1)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            results = dict(pool.imap_unordered(work, tasks))
    else:
        results = dict(map(work, tasks))

    names = ["base"] + [c.name for c in configs]
    method_of = {"base": "base", **{c.name: c.method for c in configs}}
    metrics, selected = {}, {}
    for li, lv in enumerate(levels):
        L = level_name(lv)
        rows, srows = [], []
        metrics[L] = {}
        for n in names:
            per = [results[(li, h)]["stats"][n] for h in range(H)]
            mt = {}
            for g in ("all", "tune", "shift"):
                c = np.array([p[g]["cov"] for p in per]); w = np.array([p[g]["wid"] for p in per])
                k = np.array([p[g]["n"] for p in per])
                mt[g] = {"coverage": float(c.sum() / max(k.sum(), 1)), "width": float(w.sum() / max(k.sum(), 1)),
                         "per_horizon_coverage": (c / np.maximum(k, 1)).round(4).tolist(),
                         "per_horizon_width": (w / np.maximum(k, 1)).round(3).tolist(),
                         "local_coverage_error": float(np.mean([p[g]["local_err"] for p in per])),
                         "frac_vm_days_10pp_under": float(np.mean([p[g]["vmday_under10"] for p in per]))}
                rows.append({"config": n, "method": method_of[n], "group": g,
                             **{kk: vv for kk, vv in mt[g].items() if not kk.startswith("per_")}})
            sh = {}
            for wname in ("before_1d", "after_1d", "after_all"):
                c = sum(results[(li, h)]["shift"][n][wname]["cov"] for h in range(H))
                w = sum(results[(li, h)]["shift"][n][wname]["wid"] for h in range(H))
                k = sum(results[(li, h)]["shift"][n][wname]["n"] for h in range(H))
                sh[wname] = {"coverage": c / max(k, 1), "width": w / max(k, 1)}
                srows.append({"config": n, "method": method_of[n], "window": wname,
                              "coverage": c / max(k, 1), "width": w / max(k, 1)})
            mt["shift_windows"] = sh
            metrics[L][n] = mt
        table = pd.DataFrame(rows)
        table.to_csv(out / f"table_{L.rstrip('%')}.csv", index=False)
        pd.DataFrame(srows).to_csv(out / f"shift_table_{L.rstrip('%')}.csv", index=False)
        # selection on the tuning VMs only
        tune = table[table.group == "tune"].copy()
        sel = {"base": "base"}
        for meth in [m for m in cfg["methods"]]:
            cand = tune[tune.method == meth].copy()
            cand["gap"] = (cand.coverage - lv).abs()
            ok = cand[cand.gap <= cfg["coverage_tolerance"]]
            sel[meth] = (ok.sort_values("width") if len(ok) else cand.sort_values("gap")).iloc[0].config
        selected[L] = sel
        # traces for figures
        sel_names = list(dict.fromkeys(sel.values()))
        tr = {"ts": data["ts"], "shift_step": shift_step[shift_step >= 0],
              "shift_vm_ids": data["vm_ids"][shift_step >= 0], "methods": np.array(list(sel.keys())),
              "configs": np.array(sel_names), "plot_horizons": np.array(cfg["plot_horizons"]),
              "eval_start": eval_start, "pre_min_step": pre_min}
        for n in sel_names:
            tr[f"time_cov__{n}"] = np.stack([results[(li, h)]["time"][n][0] for h in range(H)])
            tr[f"time_wid__{n}"] = np.stack([results[(li, h)]["time"][n][1] for h in range(H)])
            for hp in cfg["plot_horizons"]:
                if hp <= H:
                    c, w, o = results[(li, hp - 1)]["trace"][n]
                    tr[f"shift_cov__{n}__h{hp}"], tr[f"shift_wid__{n}__h{hp}"] = c, w
                    tr[f"shift_obs__h{hp}"] = o
        np.savez_compressed(out / f"traces_{L.rstrip('%')}.npz", **tr)

    meta = {"predictions": str(pred_path), "n_vms": int(V), "n_steps": int(T), "horizons": int(H),
            "n_shift_vms": int((shift_step >= 0).sum()),
            "n_shift_vms_with_1d_before": int((shift_step >= pre_min).sum()),
            "eval_start_day": eval_start / STEPS_PER_DAY,
            "config": cfg, "runtime_s": round(time.time() - t0)}
    json.dump({"meta": meta, "selected": selected, "metrics": metrics}, open(out / "metrics.json", "w"), indent=1)
    json.dump(selected, open(out / "selected.json", "w"), indent=1)
    for L, sel in selected.items():
        print(f"\n{L} interval (selected on tune VMs; coverage / width)")
        for meth, n in sel.items():
            m = metrics[L][n]
            sw = m["shift_windows"]
            print(f"  {meth:6s} {n:22s} all {m['all']['coverage']:.3f}/{m['all']['width']:.2f}  "
                  f"shift {m['shift']['coverage']:.3f}/{m['shift']['width']:.2f}  "
                  f"before {sw['before_1d']['coverage']:.3f} after1d {sw['after_1d']['coverage']:.3f}  "
                  f"local err {m['all']['local_coverage_error']:.3f}")
    from .plots import make_all
    make_all(out)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
