# Robust Time-Series Forecasting Under Distribution Shift

MSML612 team project: a transformer forecaster (quantile PatchTST) for cloud VM CPU
utilization, wrapped with online conformal calibration so prediction intervals recover
their coverage after a distribution shift.

- **Data:** Azure Public Dataset V2 (2019 VM trace), 5-minute CPU readings, 30 days.
  See [docs/dataset.md](docs/dataset.md).
- **Baseline:** PatchTST (Nie et al., ICLR 2023) with a non-crossing multi-quantile head,
  trained with pinball loss. Vendored from the official repo (Apache-2.0).
- **Contribution (next phase):** online conformal wrappers (ACI, quantile tracking,
  conformal PID) from `aangelopoulos/conformal-time-series` (MIT) applied to the
  baseline's own quantiles (CQR scores). See [docs/implementations.md](docs/implementations.md).
- **Citations:** [CITATIONS.bib](CITATIONS.bib).

## Layout

```
configs/            run configs (small CPU run, full Zaratan run)
scripts/            fetch_azure2019.py -> build_dataset.py -> explore_data.py
src/msml612/
  data/windows.py   chronological splits + sliding windows
  models/           quantile_patchtst.py (ours) + patchtst/ (vendored, unmodified)
  conformal/        online conformal wrappers (next phase)
  train.py          training loop (pinball loss, early stopping)
  evaluate.py       test metrics, rolling forecasts for the conformal phase, coverage plot
  metrics.py
slurm/              Zaratan batch jobs
docs/               dataset + implementation notes, EDA outputs (docs/eda)
data/, outputs/     generated locally, git-ignored
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build on Zaratan
pip install -r requirements.txt
```

The fetch script needs `bash`, `curl`, `gzip` and `awk`. On Windows it uses Git Bash
(`C:/Program Files/Git/bin/bash.exe`, override with `BASH_EXE`).

## Pipeline

```bash
# 1. download vmtable, pick 2000 full-month VMs, stream-filter the 195 reading files (~156 GB read, ~2 GB kept)
python scripts/fetch_azure2019.py --out data/raw/azure2019 --workers 8
# 2. dense [n_vm, 8639, 3] array + 100-VM sample
python scripts/build_dataset.py --raw data/raw/azure2019 --out data/azure2019_cpu_2000vm.npz \
    --sample 100 --sample-out data/sample/azure2019_cpu_sample100.npz
# 3. exploration and natural-shift candidates
python scripts/explore_data.py --data data/azure2019_cpu_2000vm.npz --out docs/eda
# 4. baseline (set PYTHONPATH=src)
python -m msml612.train --config configs/baseline_cpu_small.yaml
python -m msml612.evaluate --run outputs/baseline_cpu_small
```

## Results so far

See [docs/baseline_results.md](docs/baseline_results.md).

## What runs where

See [docs/zaratan.md](docs/zaratan.md).
