#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

OUTPUT_DIR="./outputs/hrrr_era5"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/corrdiff"

CONFIG_FILE_CORRDIFF_REG="config_training_corrdiff_regression.yaml"
CONFIG_FILE_CORRDIFF_DIFF="config_training_corrdiff_diffusion.yaml"

CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_CORRDIFF_REG bash scripts/01_train_regression.sh
CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_CORRDIFF_DIFF bash scripts/04_train_diffusion.sh
