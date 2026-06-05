#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/uq_rmse"
REG_DIR="${OUTPUT_DIR}/checkpoints/rematch_u"

CONFIG_FILE_UQ="config_training_uq_rmse.yaml"
CONFIG_FILE_DIFF="config_training_corrdiff_diffusion.yaml"

GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR REG_DIR=$REG_DIR CONFIG=$CONFIG_FILE_UQ bash scripts/02_train_uq.sh
GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR REG_DIR=$REG_DIR CONFIG=$CONFIG_FILE_DIFF bash scripts/04_train_diffusion_from_regression_uq.sh
