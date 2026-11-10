# What runs locally vs. on Zaratan

A typical development laptop (no CUDA GPU) is fine for data preparation, debugging and
small CPU runs; the full-size baseline and the experiment sweeps run on the UMD Zaratan cluster.

| Step | Where | Why |
|---|---|---|
| `fetch_azure2019.py` (156 GB streamed, ~2 GB kept) | either | Network-bound. Done once locally; on Zaratan use `slurm/fetch_data.sbatch` if the team wants the data there instead of copying `data/azure2019_cpu_2000vm.npz` over with `scp`/Globus. |
| `build_dataset.py`, `explore_data.py` | local | Minutes on CPU. |
| `configs/baseline_cpu_small.yaml` (200 VMs, d_model 64) | local | Debugging and quick sanity numbers, tens of minutes on CPU. |
| `configs/baseline_zaratan.yaml` (all VMs, paper-sized PatchTST, stride-1 windows) | **Zaratan GPU** | ~15M training windows per pass; hours on one GPU, impractical on CPU. `slurm/train_baseline.sbatch`. |
| `evaluate --stride 1` on the full model | **Zaratan GPU** | Rolling test forecasts for every VM and step (input to the conformal phase). |
| Online conformal replay (ACI / quantile tracker / PID over saved forecasts) | local or Zaratan CPU | Vectorised over VMs, a few minutes on 4-8 cores. `slurm/conformal.sbatch` (CPU partition). |
| Seeds x shift types x conformal methods sweep, extra baselines (vanilla Transformer, EnbPI) | **Zaratan** (job array) | Many independent runs. |

## Setup on Zaratan

```bash
module load python
python -m venv .venv && source .venv/bin/activate   # the sbatch scripts activate the repo's .venv
pip install torch            # CUDA build
pip install -r requirements.txt
mkdir -p logs                # Slurm does not create the log folder
```

The scripts use the class allocation (`--account=msml612-class`). GPU jobs must name the GPU
type (`--gres=gpu:a100:1`); a plain `gpu:1` is rejected. Each job appends a START and a
DONE/FAIL line to `logs/job_history.log`. To get e-mail notifications, add
`--mail-type=BEGIN,END,FAIL --mail-user=<your e-mail>` to the `sbatch` command.
