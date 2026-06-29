#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/convfno"

CONFIG_FILE="config_training_convfno_regression.yaml"

GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE bash scripts/01_train_regression_convfno.sh
