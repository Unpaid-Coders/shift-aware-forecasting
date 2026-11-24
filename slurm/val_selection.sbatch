#!/bin/bash
# Shift-aware hyperparameter selection on the validation days, then evaluation on the test shifts.
# 1. Inject a variance shift into the validation days (one step per VM in day 19.5-20), forecast the
#    validation split with the trained baseline (GPU), and pick one setting per method on it
#    (coverage overall and in the day after the shift, then width; burn-in day 18.0-18.5).
# 2. Rerun the conformal methods with those settings on the clean test forecasts and on each
#    injected test shift that has forecasts (outputs/conformal_*_valsel).
# Submit from the repo root after `mkdir -p logs`. Needs outputs/baseline_zaratan/best.pt and the
# test forecasts from slurm/injected_shifts.sbatch.
#SBATCH --job-name=val-selection
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
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
SCALE=${SCALE:-3.0}               # variance factor of the validation shift
VAL=outputs/conformal_shift_scale_val
WORKERS=${SLURM_CPUS_PER_TASK:-8}

echo "=== validation: inject, forecast, select ==="
python -m shift_aware_forecasting.data.inject --data data/azure2019_cpu_2000vm.npz --kind scale \
    --scale "$SCALE" --first-day 19.5 --last-day 19 --out data/injected_val
python -m shift_aware_forecasting.evaluate --run "$RUN" --split val --stride 1 --tag shift_scale_val \
    --data data/injected_val/azure2019_cpu_2000vm_scale.npz
python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml \
    --predictions "$RUN/predictions_val_s1_shift_scale_val.npz" --out "$VAL" \
    --shift-csv data/injected_val/shifts_scale.csv --select-on shift --burnin-days 0.5 \
    --shift-label "an injected variance shift (validation)" --workers "$WORKERS"

echo "=== test: clean data with the validation-selected settings ==="
python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml \
    --predictions "$RUN/predictions_test_s1.npz" --out outputs/conformal_zaratan_valsel \
    --selected "$VAL/selected.json" --workers "$WORKERS"

declare -A LABEL=([level]="an injected level shift" [scale]="an injected variance shift"
                  [level_scale]="an injected level and variance shift" [scale_x1.5]="an injected 1.5x variance shift")
for NAME in level scale level_scale scale_x1.5; do
    PRED="$RUN/predictions_test_s1_shift_$NAME.npz"
    KIND=${NAME%_x*}; SUFFIX=${NAME#"$KIND"}
    if [ ! -f "$PRED" ]; then echo "skip $NAME (no $PRED)"; continue; fi
    echo "=== test: $NAME with the validation-selected settings ==="
    python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml \
        --predictions "$PRED" --out "outputs/conformal_shift_${NAME}_valsel" \
        --shift-csv "data/injected$SUFFIX/shifts_$KIND.csv" --selected "$VAL/selected.json" \
        --shift-label "${LABEL[$NAME]}" --workers "$WORKERS"
done
