#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_training_rematchu_regression.yaml}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/checkpoints/uq_rmse}"
REG_DIR="${REG_DIR:-/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u}"

REG_PATH=$(ls -1 "$REG_DIR/checkpoints_regression"/UNet.0.*.mdlus \
  | sort -V \
  | tail -n 1)

echo "Running uncertainty quantification network training - rmse"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
  -m rematch.train_uq_rmse --config-name=$CONFIG \
  ++training.io.regression_checkpoint_path=$REG_PATH \
  ++training.io.checkpoint_dir=$CHECKPOINTS_DIR 

# Bias corrector 
# SAVE_DIR="./checkpoints/hrrr_mini_east_train/bias_corrector/diffusion_2020_GT_rmse"
# BIAS_CORRECTOR_CHECKPOINT="./checkpoints/hrrr_mini_east_train/bias_corrector/2020_rmse_q90/checkpoints_regression/CorrDiffRegressionUNet.0.3000000.mdlus"
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29501 --nproc_per_node=2 \
#   train_bias_corrector.py --config-name=config_training_hrrr_mini_east_diffusion_temporal.yaml \
#   ++training.io.regression_checkpoint_path=$LATER_REGRESSION \
#   ++training.io.bias_checkpoint_path=$BIAS_CORRECTOR_CHECKPOINT \
#   ++training.io.checkpoint_dir=$SAVE_DIR \
#   ++training.hp.training_duration=8000000 \
#   ++training.io.save_checkpoint_freq=1000000 \
#   ++dataset.time_flag=False \
#   ++training.hp.total_batch_size=128