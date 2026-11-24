#!/bin/bash
# Offline split-conformal baseline (conformal.split): fixed offsets calibrated on the clean
# validation days, evaluated next to the online methods on the clean test data and on each
# injected test shift that has forecasts (outputs/conformal_*_split).
# 1. Forecast the clean validation split with the trained baseline (GPU; skipped if the file exists).
# 2. Calibrate one offset per level and horizon (pooled over VMs, and per VM).
# 3. Rerun conformal.run with those offsets; the online methods reuse the validation-selected
#    settings from slurm/val_selection.sbatch, so their numbers match outputs/conformal_*_valsel.
# Submit from the repo root after `mkdir -p logs`. Needs outputs/baseline_zaratan/best.pt, the
# test forecasts from slurm/injected_shifts.sbatch and outputs/conformal_shift_scale_val/selected.json.
#SBATCH --job-name=split-conformal
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
SEL=${SEL:-outputs/conformal_shift_scale_val/selected.json}
OFFSETS=$RUN/split_offsets.npz
WORKERS=${SLURM_CPUS_PER_TASK:-8}

echo "=== validation: clean forecasts and split-conformal calibration ==="
if [ ! -f "$RUN/predictions_val_s1.npz" ]; then
    python -m shift_aware_forecasting.evaluate --run "$RUN" --split val --stride 1
fi
python -m shift_aware_forecasting.conformal.split --config configs/conformal_zaratan.yaml \
    --predictions "$RUN/predictions_val_s1.npz" --out "$OFFSETS"

echo "=== test: clean data ==="
python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml \
    --predictions "$RUN/predictions_test_s1.npz" --out outputs/conformal_zaratan_split \
    --selected "$SEL" --split-offsets "$OFFSETS" --workers "$WORKERS"

declare -A LABEL=([level]="an injected level shift" [scale]="an injected variance shift"
                  [level_scale]="an injected level and variance shift" [scale_x1.5]="an injected 1.5x variance shift")
for NAME in level scale level_scale scale_x1.5; do
    PRED="$RUN/predictions_test_s1_shift_$NAME.npz"
    KIND=${NAME%_x*}; SUFFIX=${NAME#"$KIND"}
    if [ ! -f "$PRED" ]; then echo "skip $NAME (no $PRED)"; continue; fi
    echo "=== test: $NAME ==="
    python -m shift_aware_forecasting.conformal.run --config configs/conformal_zaratan.yaml \
        --predictions "$PRED" --out "outputs/conformal_shift_${NAME}_split" \
        --shift-csv "data/injected$SUFFIX/shifts_$KIND.csv" --selected "$SEL" --split-offsets "$OFFSETS" \
        --shift-label "${LABEL[$NAME]}" --workers "$WORKERS"
done
