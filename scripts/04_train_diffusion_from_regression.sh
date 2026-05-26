#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_training_corrdiff_diffusion.yaml}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/checkpoints/corrdiff}"


echo "Running diffusion from regression model"
REG_PATH=$(ls -1 "$CHECKPOINTS_DIR/checkpoints_regression"/UNet.0.*.mdlus \
| sort -V \
| tail -n 1)
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
-m rematch.train --config-name=$CONFIG \
++training.io.checkpoint_dir=$CHECKPOINTS_DIR \
++training.io.regression_checkpoint_path=$REG_PATH

