#!/bin/bash
# Generic DSUB job script for a single paper-aligned ZuCo task unit.
#
# Usage:
#   TASK=ZuCoSR dsub -s scripts/dsub_zuco_threshold_sweep_and_train.sh
#   TASK=ZuCoNR RUN_FULL=0 dsub -s scripts/dsub_zuco_threshold_sweep_and_train.sh

#DSUB -n zuco_task_threshold_sweep
#DSUB -N 1
#DSUB -A root.project.P23Z10200N0876
#DSUB -R "cpu=16;gpu=1;mem=120000"
#DSUB -oo /home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/log/submit/%J.zuco_task_threshold_sweep.out
#DSUB -eo /home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26/log/submit/%J.zuco_task_threshold_sweep.err

set -euo pipefail

TASK="${TASK:?Set TASK=ZuCoSR|ZuCoNR|ZuCoTSR before submit}"
JOB_PATH="${JOB_PATH:-/home/share/huadjyin/home/sunmengmeng/work/EEG/BrainMosaic_ICLR26}"
LOG_DIR="${LOG_DIR:-$JOB_PATH/log/${TASK}_threshold_sweep}"
SUBMIT_LOG_DIR="$JOB_PATH/log/submit"
mkdir -p "$LOG_DIR" "$SUBMIT_LOG_DIR"

HOST_SHORT="$(hostname -s)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/${TASK}_threshold_sweep_${RUN_TS}_${HOST_SHORT}.log"

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module load "${GCC_MODULE:-compilers/gcc/9.3.0}"
module load "${CUDA_MODULE:-compilers/cuda/11.8}"
module load "${NCCL_MODULE:-libs/nccl/2.16.5-1_cuda11.8}"
module load "${CUDNN_MODULE:-libs/cudnn/8.8.1_cuda11}"
module load "${OPENBLAS_MODULE:-libs/openblas/0.3.26_gcc9.3.0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1

CONDA_SH="${CONDA_SH:-/home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-BrainMosaic}"
PYTHON_BIN="${PYTHON_BIN:-/home/share/huadjyin/home/sunmengmeng/.conda/envs/BrainMosaic/bin/python}"
PIPELINE_SCRIPT="$JOB_PATH/scripts/run_zuco_threshold_sweep_and_train.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

TAG="$(echo "$TASK" | tr '[:upper:]' '[:lower:]')"
export THRESHOLDS="${THRESHOLDS:-0.78 0.85 0.90 0.95}"
export SMOKE_EPOCHS="${SMOKE_EPOCHS:-5}"
export SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
export RUN_FULL="${RUN_FULL:-1}"
export FULL_MODE="${FULL_MODE:-baseline_and_best}"
export BASELINE_THRESHOLD="${BASELINE_THRESHOLD:-0.78}"
export FULL_EPOCHS="${FULL_EPOCHS:-50}"
export FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
export FULL_NUM_WORKERS="${FULL_NUM_WORKERS:-4}"
export BASE_TRAIN_CONFIG="${BASE_TRAIN_CONFIG:-configs/train.${TAG}.json}"
export BASE_TOKEN_CONFIG="${BASE_TOKEN_CONFIG:-configs/token_bank.${TAG}.json}"
export SWEEP_ROOT="${SWEEP_ROOT:-outputs/${TAG}_threshold_sweep}"
export FULL_ROOT="${FULL_ROOT:-outputs/${TAG}_full}"

{
  echo "[INFO] start time: $(date '+%F %T')"
  echo "[INFO] host: $HOST_SHORT"
  echo "[INFO] task: $TASK"
  echo "[INFO] job path: $JOB_PATH"
  echo "[INFO] conda env: $CONDA_ENV"
  echo "[INFO] python: $PYTHON_BIN"
  echo "[INFO] pipeline script: $PIPELINE_SCRIPT"
  echo "[INFO] thresholds: $THRESHOLDS"
  echo "[INFO] sweep root: $SWEEP_ROOT"
  echo "[INFO] full root: $FULL_ROOT"
  module list
} 2>&1 | tee -a "$RUN_LOG"

(
  cd "$JOB_PATH"
  bash "$PIPELINE_SCRIPT" "$TASK"
) 2>&1 | tee -a "$RUN_LOG"

echo "[INFO] end time: $(date '+%F %T')" | tee -a "$RUN_LOG"
echo "[INFO] run log: $RUN_LOG" | tee -a "$RUN_LOG"
