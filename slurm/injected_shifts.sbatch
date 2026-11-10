#!/bin/bash
# Injected-shift experiment: for each shift kind, inject shifts into the test days, forecast them with
# the trained baseline (GPU), then run the online conformal methods with the settings selected on the
# clean data. Submit from the repo root after `mkdir -p logs`. Needs outputs/baseline_zaratan/best.pt
# and outputs/conformal_zaratan/selected.json (from slurm/conformal.sbatch).
#SBATCH --job-name=injected-shifts
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

RUN=${RUN:-outputs/baseline_zaratan}
KINDS=${KINDS:-"level scale level_scale"}
declare -A LABEL=([level]="an injected level shift" [scale]="an injected variance shift"
                  [level_scale]="an injected level and variance shift")
for KIND in $KINDS; do
    echo "=== $KIND ==="
    python -m shift_aware_forecasting.data.inject --data data/azure2019_cpu_2000vm.npz --kind "$KIND" \
        --out data/injected
    python -m shift_aware_forecasting.evaluate --run "$RUN" --stride 1 --tag "shift_$KIND" \
        --data "data/injected/azure2019_cpu_2000vm_$KIND.npz"
    python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml \
        --predictions "$RUN/predictions_test_s1_shift_$KIND.npz" --out "outputs/conformal_shift_$KIND" \
        --shift-csv "data/injected/shifts_$KIND.csv" --selected outputs/conformal_zaratan/selected.json \
        --shift-label "${LABEL[$KIND]}" --workers "${SLURM_CPUS_PER_TASK:-8}"
done
