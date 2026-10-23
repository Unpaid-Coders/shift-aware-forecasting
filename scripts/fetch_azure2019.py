"""Download the Azure Public Dataset V2 (2019) VM trace and keep a subset of VMs.

The CPU readings are split into 195 gzipped CSV files (~856 MB each, ~156 GB total)
ordered by time, so every file holds ~3.7 hours of readings for *all* VMs. To get the
full 30-day series for a set of VMs we stream each file through `gzip -dc | awk` and
keep only rows whose VM id is in the selection (awk hash lookup; ~12x faster than
grep -F with 2000 patterns). Nothing large is written to disk.

Steps:
  1. download vmtable.csv.gz (~440 MB) and pick VMs (default: alive for the whole
     30 days, category Interactive / Delay-insensitive, avg CPU >= 2%, 1000 of each).
  2. stream the 195 reading files in parallel, writing filtered parts to <out>/parts.

Usage:
  python scripts/fetch_azure2019.py --out /scratch/.../azure2019 --per-category 1000 --workers 4

Data license: CC-BY 4.0 (Microsoft). Cite Cortez et al., SOSP 2017 (Resource Central).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2"
N_FILES = 195
T_END = 2591400  # last 5-min timestamp (seconds) in the 30-day trace
# The streaming filter needs bash + curl + gzip + awk. On Windows use Git Bash (not WSL's
# bash.exe, which sees paths differently); override with $BASH_EXE.
_GIT_BASH = "C:/Program Files/Git/bin/bash.exe"
BASH = os.environ.get("BASH_EXE") or (_GIT_BASH if os.name == "nt" else shutil.which("bash") or "bash")
VMTABLE_COLS = ["vm_id", "sub_id", "dep_id", "t_created", "t_deleted", "max_cpu",
                "avg_cpu", "p95_max_cpu", "category", "core_bucket", "mem_bucket"]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def select_vms(vmtable: pd.DataFrame, per_category: int, min_avg_cpu: float, seed: int) -> pd.DataFrame:
    full = vmtable[(vmtable.t_created <= 0) & (vmtable.t_deleted >= T_END)
                   & (vmtable.category != "Unknown") & (vmtable.avg_cpu >= min_avg_cpu)]
    return pd.concat([full[full.category == c].sample(per_category, random_state=seed)
                      for c in ["Interactive", "Delay-insensitive"]])


def fetch_part(i: int, ids_file: Path, parts_dir: Path, retries: int = 4) -> int:
    out = parts_dir / f"part-{i:03d}.csv"
    done = out.with_suffix(".csv.done")
    if done.exists():
        return int(done.read_text())
    url = f"{BASE}/trace_data_vm_cpu_readings_vm_cpu_readings-file-{i}-of-{N_FILES}.csv.gz"
    awk = "awk -F, 'NR==FNR{a[$1];next} ($2 in a)'"
    cmd = f"curl -sSfL --retry 3 '{url}' | gzip -dc | {awk} '{ids_file.as_posix()}' - > '{out.as_posix()}.tmp'"
    for attempt in range(retries):
        # pipefail makes curl/gzip failures visible
        r = subprocess.run([BASH, "-o", "pipefail", "-c", cmd])
        if r.returncode == 0:
            Path(f"{out}.tmp").rename(out)
            n = sum(1 for _ in open(out))
            done.write_text(str(n))
            return n
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed to fetch file {i}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=1000)
    ap.add_argument("--min-avg-cpu", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=612)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--files", type=str, default=f"1-{N_FILES}", help="range like 1-195 (for testing)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out / "parts"
    parts_dir.mkdir(exist_ok=True)

    vt_gz = args.out / "vmtable.csv.gz"
    download(f"{BASE}/trace_data_vmtable_vmtable.csv.gz", vt_gz)
    vt = pd.read_csv(vt_gz, header=None, names=VMTABLE_COLS)
    vt.to_parquet(args.out / "vmtable.parquet")

    sel_path = args.out / "selected_vms.parquet"
    if sel_path.exists():
        sel = pd.read_parquet(sel_path)
    else:
        sel = select_vms(vt, args.per_category, args.min_avg_cpu, args.seed)
        sel.to_parquet(sel_path)
    ids_file = args.out / "selected_ids.txt"
    ids_file.write_text("\n".join(sel.vm_id) + "\n")
    print(f"selected {len(sel)} VMs: {sel.category.value_counts().to_dict()}")

    lo, hi = map(int, args.files.split("-"))
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch_part, i, ids_file, parts_dir): i for i in range(lo, hi + 1)}
        for k, f in enumerate(as_completed(futs), 1):
            print(f"[{k}/{len(futs)}] file {futs[f]}: {f.result()} rows", flush=True)


if __name__ == "__main__":
    sys.exit(main())
