#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/cdm"

CONFIG_FILE_CORRDIFF_DIFF="config_training_corrdiff_diffusion.yaml"

GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_CORRDIFF_DIFF bash scripts/01_train_diffusion.sh
