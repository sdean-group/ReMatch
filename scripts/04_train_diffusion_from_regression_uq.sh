#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONFIG="${CONFIG:-config_training_corrdiff_diffusion.yaml}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"

REG_DIR="${REG_DIR:-/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/checkpoints/uq_rmse}"
UQ="${UQ:-rmse}"

echo "Running diffusion from regression model"
echo "REG_DIR=${REG_DIR}"
echo "CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
echo "CONFIG=${CONFIG}"
echo "GPUS=${GPUS}"
echo "NPROC=${NPROC}"

REG_PATH=$(
  ls -1 "${REG_DIR}/checkpoints_regression"/UNet.0.*.mdlus 2>/dev/null \
  | sort -V \
  | tail -n 1
)

UQ_PATH=$(
  ls -1 "${CHECKPOINTS_DIR}/checkpoints_regression"/UNet.0.*.mdlus 2>/dev/null \
  | sort -V \
  | tail -n 1
)

if [ -z "${REG_PATH}" ]; then
  echo "ERROR: regression checkpoint not found in ${REG_DIR}/checkpoints_regression"
  exit 1
fi

if [ -z "${UQ_PATH}" ]; then
  echo "ERROR: UQ/bias checkpoint not found in ${CHECKPOINTS_DIR}/checkpoints_regression"
  exit 1
fi

echo "Using regression checkpoint: ${REG_PATH}"
echo "Using UQ/bias checkpoint: ${UQ_PATH}"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m rematch.train_uq\
  --config-name="${CONFIG}" \
  ++hydra.job.name="uq_quantiles_diffusion"\
  "++training.io.checkpoint_dir=${CHECKPOINTS_DIR}" \
  "++training.io.regression_checkpoint_path=${REG_PATH}" \
  "++training.io.bias_checkpoint_path=${UQ_PATH}"\
  ++uq.type=$UQ