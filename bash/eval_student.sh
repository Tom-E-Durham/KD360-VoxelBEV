#!/bin/bash
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --time=2-23:59:59
#SBATCH --gres=gpu:1
#SBATCH -p gpu-bigmem
#SBATCH --job-name=eval_student

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v module >/dev/null 2>&1 && [[ -n "${CUDA_MODULE:-}" ]]; then
  module load "${CUDA_MODULE}" || true
fi

if [[ -n "${CONDA_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "$CKPT_PATH" ]]; then
  echo "ERROR: set CKPT_PATH to student checkpoint path."
  echo "Example: CKPT_PATH=/path/to/student.pth bash bash/eval_student.sh"
  exit 1
fi

MODE="${MODE:-eval}"                  # eval | speed
DATASET_NAME="${DATASET_NAME:-Dur360BEV}"
DATASET_VERSION="${DATASET_VERSION:-extended}"  # extended | mini
ENCODER_TYPE="${ENCODER_TYPE:-effb0_ori}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NWORKERS="${NWORKERS:-6}"
MAP_R="${MAP_R:-100}"
MAP_SCALE="${MAP_SCALE:-2}"
LOG_FREQ="${LOG_FREQ:-1}"
IMG_FREQ="${IMG_FREQ:-10}"
SAVE_PRED="${SAVE_PRED:-False}"
PRED_DIR="${PRED_DIR:-./results_dur360bev/equi/pred}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-20}"

if [[ "$DATASET_VERSION" != "extended" && "$DATASET_VERSION" != "mini" ]]; then
  echo "ERROR: DATASET_VERSION must be 'extended' or 'mini'."
  exit 1
fi

CMD=(
  python -u "$ROOT_DIR/eval_student_unified.py"
  --mode="$MODE"
  --batch_size="$BATCH_SIZE"
  --nworkers="$NWORKERS"
  --encoder_type="$ENCODER_TYPE"
  --checkpoint_dir="$CKPT_PATH"
  --dataset_name="$DATASET_NAME"
  --dataset_version="$DATASET_VERSION"
  --map_r="$MAP_R"
  --map_scale="$MAP_SCALE"
  --log_freq="$LOG_FREQ"
  --img_freq="$IMG_FREQ"
  --save_pred="$SAVE_PRED"
  --pred_dir="$PRED_DIR"
  --warmup="$WARMUP"
  --repeats="$REPEATS"
)

if [[ -n "${DUR360BEV_DATASET_DIR:-}" ]]; then
  CMD+=(--dataset_dir="$DUR360BEV_DATASET_DIR")
fi

"${CMD[@]}"
