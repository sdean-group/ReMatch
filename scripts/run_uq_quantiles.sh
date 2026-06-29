#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/uq_quantiles"
# REG_DIR="${OUTPUT_DIR}/checkpoints/rematch_u"

REG_DIR="${OUTPUT_DIR}/checkpoints/uq_regression"
CONFIG_FILE_REG="config_training_uq_regression.yaml"

CONFIG_FILE_UQ="config_training_uq.yaml"
CONFIG_FILE_DIFF="config_training_corrdiff_diffusion.yaml"
UQ=quantiles

# GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR REG_DIR=$REG_DIR CONFIG=$CONFIG_FILE_UQ UQ=$UQ bash scripts/02_train_uq.sh
GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR REG_DIR=$REG_DIR CONFIG=$CONFIG_FILE_DIFF UQ=$UQ bash scripts/04_train_diffusion_from_regression_uq.sh
