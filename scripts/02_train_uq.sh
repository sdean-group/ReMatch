#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_training_rematchu_regression.yaml}"
GPUS="${GPUS:-0,1}"
REG_DIR="${REG_DIR:-/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u}"

UQ="${UQ:-rmse}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/checkpoints/uq_rmse}"
REG_DIR="${REG_DIR:-/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u}"


REG_PATH=$(ls -1 "$REG_DIR/checkpoints_regression"/UNet.0.*.mdlus \
  | sort -V \
  | tail -n 1)

echo "Running uncertainty quantification network training "

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
  -m rematch.train_uq --config-name=$CONFIG \
  ++training.io.regression_checkpoint_path=$REG_PATH \
  ++training.io.checkpoint_dir=$CHECKPOINTS_DIR \
  ++uq.type=$UQ 
