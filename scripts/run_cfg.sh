#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs
CORRDIFF_CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/corrdiff"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/cfg"

CONFIG_FILE_CORRDIFF_REG="config_training_corrdiff_regression.yaml"
CONFIG_FILE_CORRDIFF_DIFF="config_training_corrdiff_diffusion.yaml"

mkdir -p $CHECKPOINTS_DIR/checkpoints_regression
REG_PATH=$(ls -1 "$CORRDIFF_CHECKPOINTS_DIR/checkpoints_regression"/UNet.0.*.mdlus \
| sort -V \
| tail -n 1)
cp $REG_PATH $CHECKPOINTS_DIR/checkpoints_regression

GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_CORRDIFF_DIFF bash scripts/04_train_diffusion_from_regression_cfg.sh
