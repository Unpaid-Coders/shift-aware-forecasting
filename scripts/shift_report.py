"""Print a text summary of the injected-shift experiment (slurm/injected_shifts.sbatch).

For the clean run and each injected shift kind: base forecaster accuracy and interval coverage,
the selected conformal settings with coverage/width overall and in the 1-day windows around the
shift, and coverage/width in hour bins aligned to the shift (from traces_<level>.npz).
Run from the repo root on the machine that has outputs/:

  python scripts/shift_report.py | tee logs/shift_report.txt
"""
import json
from pathlib import Path

import numpy as np

RUN = Path("outputs/baseline_zaratan")
KINDS = ["level", "scale", "level_scale"]
BINS = [(-24, 0), (0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72)]  # hours since shift

print("##### base forecaster (median + raw quantile intervals), clean vs injected")
for tag in ["test_s1"] + [f"test_s1_shift_{k}" for k in KINDS]:
    p = RUN / f"metrics_{tag}.json"
    if not p.exists():
        print(tag, "MISSING"); continue
    m = json.load(open(p))
    flat = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
    print(tag, json.dumps(flat))
    for k, v in m.items():
        if isinstance(v, dict):
            print("   ", k, json.dumps({kk: vv for kk, vv in v.items() if not isinstance(vv, list)}))

for out in ["outputs/conformal_zaratan"] + [f"outputs/conformal_shift_{k}" for k in KINDS]:
    out = Path(out)
    print(f"\n##### {out}")
    if not (out / "metrics.json").exists():
        print("MISSING"); continue
    res = json.load(open(out / "metrics.json"))
    print("meta:", json.dumps({k: v for k, v in res["meta"].items() if k != "config"}))
    print("selected:", json.dumps(res["selected"]))
    for L, sel in res["selected"].items():
        print(f"-- {L}: method | all cov/wid | shift-VM cov/wid | before1d cov/wid | after1d cov/wid | afterAll cov/wid | localErr | fracVMday10ppUnder")
        for meth, n in sel.items():
            m = res["metrics"][L][n]; sw = m["shift_windows"]
            print(f"   {meth:5s} {n:22s} {m['all']['coverage']:.3f}/{m['all']['width']:.2f} | "
                  f"{m['shift']['coverage']:.3f}/{m['shift']['width']:.2f} | "
                  + " | ".join(f"{sw[w]['coverage']:.3f}/{sw[w]['width']:.2f}" for w in ("before_1d", "after_1d", "after_all"))
                  + f" | {m['all']['local_coverage_error']:.3f} | {m['all']['frac_vm_days_10pp_under']:.3f}")
            print(f"         per-h cov (shift VMs): {m['shift']['per_horizon_coverage']}")
    # coverage/width in hour bins aligned to the shift, from the saved traces
    for L in res["selected"]:
        tg = L.rstrip("%")
        tr = np.load(out / f"traces_{tg}.npz")
        if len(tr["shift_step"]) == 0:
            continue
        ss = tr["shift_step"].astype(np.int64)
        for h in tr["plot_horizons"]:
            h = int(h)
            if f"shift_obs__h{h}" not in tr.files:
                continue
            obs = tr[f"shift_obs__h{h}"]
            rel = (tr["ts"].astype(np.int64) + (h - 1))[None, :] - ss[:, None]   # target time - shift step
            print(f"   aligned {L} h={h} (cov/wid per hours-since-shift bin; n VMs={len(ss)}):")
            hdr = "      " + " ".join(f"[{a},{b})".rjust(13) for a, b in BINS)
            print(hdr)
            for c in tr["configs"]:
                c = str(c)
                cov = tr[f"shift_cov__{c}__h{h}"]; wid = tr[f"shift_wid__{c}__h{h}"].astype(np.float32)
                cells = []
                for a, b in BINS:
                    msk = obs & (rel >= a * 12) & (rel < b * 12)
                    cells.append(f"{cov[msk].mean():.3f}/{wid[msk].mean():5.2f}" if msk.any() else "-")
                print(f"      {c:22s}" + " ".join(x.rjust(13) for x in cells))
