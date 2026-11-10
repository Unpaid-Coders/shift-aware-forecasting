#!/bin/bash
# Online conformal calibration (ACI, quantile tracker, conformal PID) of the baseline's test forecasts. CPU only.
# Submit from the repo root after `mkdir -p logs`; needs outputs/baseline_zaratan/predictions_test_s1.npz
# (written by slurm/train_baseline.sbatch). Runtime is a few minutes for all 1,909 VMs.
#SBATCH --job-name=conformal
#SBATCH --account=msml612-class
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# One line per job start/end in logs/job_history.log (stdout -> logs/%x-%j.out, stderr -> logs/%x-%j.err)
HIST=logs/job_history.log
echo "$(date '+%F %T') START  job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}" >> "$HIST"
trap 'rc=$?; echo "$(date '"'"'+%F %T'"'"') $([ $rc -eq 0 ] && echo DONE || echo FAIL)   job=$SLURM_JOB_ID name=$SLURM_JOB_NAME exit=$rc see logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >> "$HIST"' EXIT
set +u   # module/activate scripts reference unset variables
module load python
source .venv/bin/activate
set -u
export PYTHONPATH=$PWD/src
CFG=${CFG:-configs/conformal_zaratan.yaml}
python -m shift_aware_forecasting.conformal.run --config "$CFG" --workers "${SLURM_CPUS_PER_TASK:-8}" ${EXTRA_ARGS:-}
