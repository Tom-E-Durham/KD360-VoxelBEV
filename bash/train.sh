#!/bin/bash
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=32g
#SBATCH --time=6-23:59:59
#SBATCH --gres=gpu:ampere:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --job-name=best_dur

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
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:50}"

LOAD_CKPT="${LOAD_CKPT:-}"
if [[ -z "$LOAD_CKPT" ]]; then
  echo "ERROR: set LOAD_CKPT to your teacher checkpoint path."
  echo "Example: LOAD_CKPT=/path/to/teacher.pth bash bash/train.sh"
  exit 1
fi
DATASET_NAME="${DATASET_NAME:-Dur360BEV}"
DATASET_VERSION="${DATASET_VERSION:-extended}"
if [[ "$DATASET_VERSION" != "extended" && "$DATASET_VERSION" != "mini" ]]; then
  echo "ERROR: DATASET_VERSION must be 'extended' or 'mini'."
  exit 1
fi

python -u "$ROOT_DIR/train_distill.py" \
  --debug=False \
  --max_iters=12000 \
  --batch_size=6 \
  --nworkers=6 \
  --lr=5e-4 \
  --do_val=True \
  --use_scheduler=True \
  --scheduler_type='onecycle' \
  --backbone='effb0_ori' \
  --load_ckpt_dir="$LOAD_CKPT" \
  --dataset_name="$DATASET_NAME" \
  --dataset_version="$DATASET_VERSION" \
  --map_scale=2 \
  --gamma=2 \
  --img_freq=20 \
  --temperature=4 \
  --dis_type='cwd' \
  --stage='stage3'
