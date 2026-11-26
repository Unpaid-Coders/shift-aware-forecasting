# Robust Time-Series Forecasting Under Distribution Shift

**When a forecast says "90% sure", is it right 90% of the time, and does it stay right when the
workload suddenly changes?**

This MSML612 team project forecasts the next hour of CPU usage for 1,909 real Azure virtual machines
as a range, and checks whether the range keeps its promise. A quantile PatchTST transformer gives
accurate forecasts, but its 90% ranges hold the true value only 85% of the time, and only 63% in the
day after a large change. A light online conformal layer on top of the frozen model brings coverage
back to 90%, and, tuned on a change planted in the validation days, recovers within hours of a change
for a few percent more width.

| 90% range, test days | Coverage, normal days | Width (CPU points) | Coverage, day after a 3x variance change |
|---|---|---|---|
| Quantile PatchTST alone | 0.851 | 5.28 | 0.630 |
| + ACI, tuned on normal days | 0.900 | 5.97 | 0.844 |
| + ACI, tuned on a planted validation change | 0.901 | 6.19 | 0.891 |

Full results: [docs/baseline_results.md](docs/baseline_results.md) and
[docs/injected_shift_results.md](docs/injected_shift_results.md).

## How it works

- **Data:** Azure Public Dataset V2 (2019 VM trace), average CPU every 5 minutes for 30 days.
  Days 1-18 train, 19-21 validation, 22-30 test. See [docs/dataset.md](docs/dataset.md).
- **Forecaster:** PatchTST (Nie et al., ICLR 2023) with a non-crossing seven-quantile head, trained
  with the pinball loss. Two days of history in, the next 12 readings out. Vendored from the
  official repository (Apache-2.0).
- **Correction layer:** online conformal methods on CQR scores of the model's own quantiles:
  adaptive conformal inference (ACI), the quantile tracker and conformal PID control, adapted from
  `aangelopoulos/conformal-time-series` (MIT). A fixed split-conformal correction calibrated on the
  validation days is the non-adaptive comparison. See [docs/implementations.md](docs/implementations.md).
- **Evaluation:** 70 VMs with natural changes in the test days, plus changes planted at a known
  time in every VM (level jump, 3x and 1.5x variance, level and variance).
- **Citations:** [CITATIONS.bib](CITATIONS.bib).

## Layout

```
configs/            run configs (small CPU run, full Zaratan run, conformal sweep)
scripts/            fetch_azure2019.py -> build_dataset.py -> explore_data.py; shift_report.py; make_synthetic.py
src/shift_aware_forecasting/
  data/windows.py   chronological splits and sliding windows
  data/inject.py    planted level / variance changes at a random time per VM
  models/           quantile_patchtst.py + patchtst/ (vendored, unmodified)
  conformal/
    methods.py      ACI, quantile tracker, conformal PID, trailing window
    run.py          sweep, setting selection, natural/planted-change evaluation, figures
    split.py        offline split-conformal baseline calibrated on the validation days
    plots.py        per-run figures
    frontier.py     width-vs-error frontier over the whole sweep
    paper_figures.py  report figures and results tables
  train.py          training loop (pinball loss, early stopping)
  evaluate.py       test metrics and rolling forecasts for the conformal phase
  metrics.py
slurm/              Zaratan batch jobs (one per experiment, see below)
docs/               dataset and implementation notes, results, EDA outputs (docs/eda)
data/, outputs/     generated, git-ignored
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build on Zaratan
pip install -r requirements.txt
```

On Zaratan, run `module load python` before creating or activating the environment; the batch
scripts do both. The fetch script needs `bash`, `curl`, `gzip` and `awk`. On Windows it uses Git
Bash (`C:/Program Files/Git/bin/bash.exe`, override with `BASH_EXE`).

## Reproducing the results

Everything below runs from the repository root. On Zaratan, run `mkdir -p logs` once first (Slurm
does not create the log folder). Each job writes `logs/<name>-<job id>.out` and `.err` and appends
a START/DONE/FAIL line to `logs/job_history.log`. Add `--mail-type=BEGIN,END,FAIL
--mail-user=<your address>` to `sbatch` to get emails.

| Step | Command | Where, time | Writes |
|---|---|---|---|
| 1. Data | `sbatch slurm/fetch_data.sbatch` | CPU, about 1 h | `data/azure2019_cpu_2000vm.npz` |
| 2. Forecaster | `sbatch slurm/train_baseline.sbatch` | A100, about 1 h | `outputs/baseline_zaratan/` (checkpoint, test forecasts) |
| 3. Online correction, natural changes | `sbatch slurm/conformal.sbatch` | CPU, about 5 min | `outputs/conformal_zaratan/` |
| 4. Planted changes (level, 3x variance, both) | `sbatch slurm/injected_shifts.sbatch` | A100, about 30 min | `outputs/conformal_shift_<kind>/` |
| 5. Milder 1.5x variance change | `KINDS=scale SCALE=1.5 SUFFIX=_x1.5 sbatch slurm/injected_shifts.sbatch` | A100, about 15 min | `outputs/conformal_shift_scale_x1.5/` |
| 6. Tuning on a planted validation change | `sbatch slurm/val_selection.sbatch` | A100, about 20 min | `outputs/conformal_shift_scale_val/`, `outputs/conformal_*_valsel/` |
| 7. One-time split-conformal comparison | `sbatch slurm/split_conformal.sbatch` | A100, about 30 min | `outputs/conformal_zaratan_split/`, `outputs/conformal_shift_*_split/` |

Steps 4-7 need step 2's checkpoint, and steps 4 onwards reuse settings from step 3 and step 6, so
run them in order (`--dependency=afterok:<job id>` chains them). Other seeds or longer training:
`CFG=<config> sbatch slurm/train_baseline.sbatch`; then
`EXTRA_ARGS="--predictions <run>/predictions_test_s1.npz --out <dir>" sbatch slurm/conformal.sbatch`.

Reports and figures, after the jobs (with the environment active and `export PYTHONPATH=$PWD/src`):

```bash
python scripts/shift_report.py scale_val scale scale_valsel scale_x1.5 scale_x1.5_valsel | tee logs/shift_report_valsel.txt
python scripts/shift_report.py scale_split scale_x1.5_split | tee logs/shift_report_split.txt
python -m shift_aware_forecasting.conformal.paper_figures --out outputs/paper_figures
python -m shift_aware_forecasting.conformal.frontier --levels 90 80 --out outputs/frontier \
    --runs outputs/conformal_zaratan="clean data" outputs/conformal_shift_scale="injected variance shift"
```

### Running locally without a GPU

A 200-VM model with one day of history (`configs/baseline_cpu_small.yaml`) trains in tens of
minutes on a laptop CPU and is enough to check the pipeline:

```bash
export PYTHONPATH=$PWD/src
python -m shift_aware_forecasting.train --config configs/baseline_cpu_small.yaml
python -m shift_aware_forecasting.evaluate --run outputs/baseline_cpu_small
```

`scripts/make_synthetic.py` writes a small synthetic dataset in the same format for smoke tests
without downloading the trace.

See [docs/zaratan.md](docs/zaratan.md) for what runs where and why.
