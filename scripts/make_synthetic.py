"""Write a small synthetic npz in the same format as build_dataset.py, for smoke tests.

Daily-periodic CPU series with noise, plus a level/variance shift at day 24 in half the VMs,
so the train -> evaluate path (and later the conformal wrapper) can be exercised without
downloading the Azure trace.

  python scripts/make_synthetic.py --out data/synthetic_small.npz --n-vm 40
"""
import argparse
from pathlib import Path

import numpy as np

SPD, T = 288, 8639


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-vm", type=int, default=40)
    ap.add_argument("--shift-day", type=float, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t = np.arange(T)
    n = args.n_vm
    base = rng.uniform(5, 40, (n, 1))
    amp = rng.uniform(2, 20, (n, 1))
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    noise = rng.uniform(1, 5, (n, 1))
    avg = base + amp * np.sin(2 * np.pi * t / SPD + phase) + noise * rng.standard_normal((n, T))
    shifted = np.arange(n) % 2 == 1
    after = t >= int(args.shift_day * SPD)
    avg[np.ix_(shifted, after)] += 15 + 2 * noise[shifted] * rng.standard_normal((shifted.sum(), after.sum()))
    avg = np.clip(avg, 0, 100)
    spread = np.abs(rng.standard_normal((n, T))) * 5
    cpu = np.stack([np.clip(avg - spread, 0, 100), np.clip(avg + spread, 0, 100), avg], -1).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, cpu=cpu, observed=np.ones((n, T), bool), timestamps=t * 300,
                        vm_ids=np.array([f"syn{i}" for i in range(n)]),
                        category=np.where(shifted, "Shifted", "Stationary"),
                        core_bucket=np.array(["2"] * n), mem_bucket=np.array(["4"] * n),
                        channels=np.array(["min_cpu", "max_cpu", "avg_cpu"]))
    print(f"wrote {args.out} cpu{cpu.shape}, {shifted.sum()} VMs shifted at day {args.shift_day}")


if __name__ == "__main__":
    main()
