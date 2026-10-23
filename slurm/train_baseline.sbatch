#!/bin/bash
# Full-size baseline on one GPU. Fill in the allocation; check GPU partition/gres with `sinfo`.
#SBATCH --job-name=qpatchtst
#SBATCH --account=<YOUR_ALLOCATION>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# module load python cuda   # or activate your conda env with requirements.txt installed
export PYTHONPATH=$PWD/src
CFG=${CFG:-configs/baseline_zaratan.yaml}
python -m msml612.train --config "$CFG" ${OVERRIDES:-}
RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['out_dir'])")
python -m msml612.evaluate --run "$RUN" --stride 12
python -m msml612.evaluate --run "$RUN" --stride 1     # rolling forecasts for the conformal phase
