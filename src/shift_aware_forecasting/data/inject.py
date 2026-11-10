"""Inject controlled distribution shifts into the test period of the CPU trace.

Each VM gets one shift at a random 5-minute step between the start of `first_day` and the end of
`last_day` (seeded, so every shift kind uses the same shift times). From that step on, all
channels (min/max/avg CPU) are transformed:

  level        x + level_sd * s            s = the VM's std of avg CPU over the training days
  scale        m + scale * (x - m)         m = the VM's mean avg CPU over the day before the shift
  level_scale  both (scale first, then level)

Values are clipped to [0, 100] CPU %. Everything before the shift is unchanged, so the trained
model sees the same history as on the clean data until the shift. The defaults keep at least one
evaluated test day before every shift (day 21 is the conformal burn-in) and three days after it.

Writes <out>/<stem>_<kind>.npz (same arrays as the input) and <out>/shifts_<kind>.csv
(vm, vm_id, category, shift_step, day, kind).

  python -m shift_aware_forecasting.data.inject --data data/azure2019_cpu_2000vm.npz \\
      --kind level --out data/injected
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288
TRAIN_DAYS = 18
AVG = 2  # channel index of avg_cpu
KINDS = ("level", "scale", "level_scale")


def shift_steps(n_vm: int, first_day: float, last_day: float, seed: int) -> np.ndarray:
    lo, hi = int(first_day * STEPS_PER_DAY), int((last_day + 1) * STEPS_PER_DAY)
    return np.random.default_rng(seed).integers(lo, hi, n_vm)


def inject(cpu: np.ndarray, steps: np.ndarray, kind: str, level_sd: float, scale: float) -> np.ndarray:
    """cpu [V, T, C] in CPU % -> shifted copy."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    out = cpu.copy()
    sd = cpu[:, :TRAIN_DAYS * STEPS_PER_DAY, AVG].std(1)
    for v, s in enumerate(steps):
        seg = out[v, s:]
        if "scale" in kind:
            m = cpu[v, max(0, s - STEPS_PER_DAY):s, AVG].mean()
            seg[:] = m + scale * (seg - m)
        if "level" in kind:
            seg += level_sd * sd[v]
    return np.clip(out, 0, 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level-sd", type=float, default=3.0, help="level shift in units of the VM's training std")
    ap.add_argument("--scale", type=float, default=3.0, help="factor applied to deviations from the pre-shift mean")
    ap.add_argument("--first-day", type=float, default=23)
    ap.add_argument("--last-day", type=float, default=26)
    ap.add_argument("--seed", type=int, default=612)
    args = ap.parse_args()

    d = dict(np.load(args.data, allow_pickle=False))
    steps = shift_steps(d["cpu"].shape[0], args.first_day, args.last_day, args.seed)
    d["cpu"] = inject(d["cpu"], steps, args.kind, args.level_sd, args.scale).astype(d["cpu"].dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.data.stem}_{args.kind}.npz", **d)
    pd.DataFrame({"vm": np.arange(len(steps)), "vm_id": d["vm_ids"], "category": d["category"],
                  "shift_step": steps, "day": steps / STEPS_PER_DAY, "kind": args.kind}
                 ).to_csv(args.out / f"shifts_{args.kind}.csv", index=False)
    print(f"{args.kind}: {len(steps)} VMs shifted between day {args.first_day:g} and day {args.last_day + 1:g}"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
