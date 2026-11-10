#!/bin/bash
# Full-size baseline on one A100. Submit from the repo root after `mkdir -p logs`; needs data/azure2019_cpu_2000vm.npz.
#SBATCH --job-name=qpatchtst
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# One line per job start/end in logs/job_history.log (stdout -> logs/%x-%j.out, stderr -> logs/%x-%j.err)
HIST=logs/job_history.log
echo "$(date '+%F %T') START  job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}" >> "$HIST"
trap 'rc=$?; echo "$(date '"'"'+%F %T'"'"') $([ $rc -eq 0 ] && echo DONE || echo FAIL)   job=$SLURM_JOB_ID name=$SLURM_JOB_NAME exit=$rc see logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >> "$HIST"' EXIT
nvidia-smi 2>/dev/null || true
set +u   # module/activate scripts reference unset variables
module load python
source .venv/bin/activate
set -u
export PYTHONPATH=$PWD/src
CFG=${CFG:-configs/baseline_zaratan.yaml}
python -m msml612.train --config "$CFG" ${OVERRIDES:-}
RUN=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['out_dir'])")
python -m msml612.evaluate --run "$RUN" --stride 12
python -m msml612.evaluate --run "$RUN" --stride 1     # rolling forecasts for the conformal phase
