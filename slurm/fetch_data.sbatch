#!/bin/bash
# Download + filter the Azure 2019 trace on Zaratan (CPU only, network-bound, ~1 h).
# Fill in the allocation and paths; check partition names with `sinfo`.
#SBATCH --job-name=azure2019-fetch
#SBATCH --account=<YOUR_ALLOCATION>
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
DATA=${DATA:-$HOME/scratch/azure2019}      # raw parts (~2 GB for 2000 VMs)
cd "$SLURM_SUBMIT_DIR"
python scripts/fetch_azure2019.py --out "$DATA" --per-category 1000 --workers 8
python scripts/build_dataset.py --raw "$DATA" --out data/azure2019_cpu_2000vm.npz
