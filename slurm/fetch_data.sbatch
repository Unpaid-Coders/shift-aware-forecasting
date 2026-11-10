#!/bin/bash
# Download + filter the Azure 2019 trace on Zaratan (CPU only, network-bound, ~1 h).
# Submit from the repo root after `mkdir -p logs` (Slurm will not create the log folder).
#SBATCH --job-name=azure2019-fetch
#SBATCH --account=msml612-class
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# One line per job start/end in logs/job_history.log (stdout -> logs/%x-%j.out, stderr -> logs/%x-%j.err)
HIST=logs/job_history.log
echo "$(date '+%F %T') START  job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}" >> "$HIST"
trap 'rc=$?; echo "$(date '"'"'+%F %T'"'"') $([ $rc -eq 0 ] && echo DONE || echo FAIL)   job=$SLURM_JOB_ID name=$SLURM_JOB_NAME exit=$rc see logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >> "$HIST"' EXIT
DATA=${DATA:-data/raw/azure2019}      # raw parts (~2 GB for 2000 VMs); override with DATA=/path
set +u   # module/activate scripts reference unset variables
module load python
source .venv/bin/activate
set -u
python scripts/fetch_azure2019.py --out "$DATA" --per-category 1000 --workers 8
python scripts/build_dataset.py --raw "$DATA" --out data/azure2019_cpu_2000vm.npz
