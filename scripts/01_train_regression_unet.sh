#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_training_rematchu_regression.yaml}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/hrrr_era5/checkpoints/rematch_u}"

echo "Running regression"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
  -m rematch.train \
  --config-name="${CONFIG}" \
  ++training.io.checkpoint_dir="${CHECKPOINTS_DIR}" 
