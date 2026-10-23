"""Turn the filtered Azure 2019 CPU reading parts into a dense [n_vm, T, 3] array.

Output (npz): cpu (float32, n_vm x T x 3 with channels min/max/avg CPU %), observed
(bool mask before gap filling), vm_ids, category, core_bucket, mem_bucket, timestamps.

Gaps up to --max-gap steps are forward-filled; VMs whose missing fraction exceeds
--max-missing are dropped. The raw mask is kept so evaluation can skip filled points.

Usage:
  python scripts/build_dataset.py --raw /path/azure2019 --out data/azure2019_cpu.npz
  python scripts/build_dataset.py ... --sample 100 --sample-out data/sample/azure2019_cpu_sample100.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEP = 300
T_END = 2591400
CHANNELS = ["min_cpu", "max_cpu", "avg_cpu"]


def load_parts(parts_dir: Path, ids: set) -> pd.DataFrame:
    frames = []
    for p in sorted(parts_dir.glob("part-*.csv")):
        df = pd.read_csv(p, header=None, names=["ts", "vm_id"] + CHANNELS)
        frames.append(df[df.vm_id.isin(ids)])  # grep -F matches substrings; enforce exact ids
    return pd.concat(frames, ignore_index=True)


def ffill_limited(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Forward-fill NaNs along axis 1 for runs of at most max_gap steps."""
    out = pd.DataFrame(x.T).ffill(limit=max_gap).bfill(limit=max_gap).to_numpy().T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="dir written by fetch_azure2019.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--max-missing", type=float, default=0.02)
    ap.add_argument("--sample", type=int, default=0, help="also write a small random subset")
    ap.add_argument("--sample-out", type=Path)
    ap.add_argument("--seed", type=int, default=612)
    args = ap.parse_args()

    sel = pd.read_parquet(args.raw / "selected_vms.parquet").reset_index(drop=True)
    df = load_parts(args.raw / "parts", set(sel.vm_id))
    df = df.drop_duplicates(["vm_id", "ts"])
    n_files = len(list((args.raw / "parts").glob("part-*.csv")))
    print(f"{len(df):,} readings from {n_files} part files")

    timestamps = np.arange(0, T_END + STEP, STEP)
    T = len(timestamps)
    vm_index = {v: i for i, v in enumerate(sel.vm_id)}
    cpu = np.full((len(sel), T, len(CHANNELS)), np.nan, dtype=np.float32)
    ri = df.vm_id.map(vm_index).to_numpy()
    ti = (df.ts.to_numpy() // STEP).astype(int)
    cpu[ri, ti] = df[CHANNELS].to_numpy(dtype=np.float32)

    observed = ~np.isnan(cpu[..., 2])
    missing = 1 - observed.mean(1)
    for c in range(len(CHANNELS)):
        cpu[..., c] = ffill_limited(cpu[..., c], args.max_gap)
    keep = (missing <= args.max_missing) & ~np.isnan(cpu).any((1, 2))
    print(f"missing fraction: median {np.median(missing):.4f}, max {missing.max():.4f}; "
          f"keeping {keep.sum()}/{len(keep)} VMs")

    meta = sel[keep].reset_index(drop=True)
    payload = dict(cpu=cpu[keep], observed=observed[keep], timestamps=timestamps,
                   vm_ids=meta.vm_id.to_numpy(str), category=meta.category.to_numpy(str),
                   core_bucket=meta.core_bucket.astype(str).to_numpy(str),
                   mem_bucket=meta.mem_bucket.astype(str).to_numpy(str), channels=np.array(CHANNELS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out} cpu{payload['cpu'].shape}")

    if args.sample:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(keep.sum(), args.sample, replace=False))
        sub = {k: (v[idx] if v.shape[:1] == (keep.sum(),) else v) for k, v in payload.items()}
        args.sample_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.sample_out, **sub)
        print(f"wrote {args.sample_out} cpu{sub['cpu'].shape}")


if __name__ == "__main__":
    main()
