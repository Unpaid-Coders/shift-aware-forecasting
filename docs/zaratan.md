# What runs locally vs. on Zaratan

This laptop has 22 logical CPUs, 15 GB RAM and no CUDA GPU, so it is fine for data prep,
debugging and small CPU runs, but the full-size baseline and the experiment sweeps belong
on Zaratan.

| Step | Where | Why |
|---|---|---|
| `fetch_azure2019.py` (156 GB streamed, ~2 GB kept) | either | Network-bound. Done once locally; on Zaratan use `slurm/fetch_data.sbatch` if the team wants the data there instead of copying `data/azure2019_cpu_2000vm.npz` over with `scp`/Globus. |
| `build_dataset.py`, `explore_data.py` | local | Minutes on CPU. |
| `configs/baseline_cpu_small.yaml` (200 VMs, d_model 64) | local | Debugging and quick sanity numbers, tens of minutes on CPU. |
| `configs/baseline_zaratan.yaml` (all VMs, paper-sized PatchTST, stride-1 windows) | **Zaratan GPU** | ~15M training windows per pass; hours on one GPU, impractical on CPU. `slurm/train_baseline.sbatch`. |
| `evaluate --stride 1` on the full model | **Zaratan GPU** | Rolling test forecasts for every VM and step (input to the conformal phase). |
| Online conformal replay (ACI / quantile tracker / PID over saved forecasts) | local or Zaratan CPU | Cheap numpy loops over `predictions_test_s1.npz`; parallelise per VM if slow. |
| Seeds x shift types x conformal methods sweep, extra baselines (vanilla Transformer, EnbPI) | **Zaratan** (job array) | Many independent runs. |

## Setup on Zaratan

```bash
module load python   # check `module avail` for a recent Python / CUDA
python -m venv ~/venvs/msml612 && source ~/venvs/msml612/bin/activate
pip install torch            # CUDA build
pip install -r requirements.txt
```

Fill in `--account` in `slurm/*.sbatch` with the class allocation and check the GPU
partition and `--gres` names with `sinfo`; both are placeholders.
