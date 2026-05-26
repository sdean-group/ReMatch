#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/rematch_u"
PCA_OT_DIR="/data/rematch_u/hrrr_era5/pca_ot"

SOURCE_DATASET=/data/rematch_u/hrrr_era5/pca_ot/reg_trainset.nc
TARGET_DATASET=/data/rematch_u/hrrr_era5/pca_ot/reg_calibrationset.nc


CONFIG_FILE_REMATCH_REG="config_training_rematchu_regression.yaml"
CONFIG_FILE_REMATCH_DIFF="config_training_rematch_diffusion.yaml"
CONFIG_FILE_GEN="config_generate.yaml"

GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_REMATCH_REG bash scripts/01_train_regression_unet.sh
SOURCE_DATASET=$SOURCE_DATASET TARGET_DATASET=$TARGET_DATASET CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_GEN PCA_OT_DIR=$PCA_OT_DIR bash scripts/02_run_pca.sh
SOURCE_DATASET=$SOURCE_DATASET TARGET_DATASET=$TARGET_DATASET PCA_OT_DIR=$PCA_OT_DIR bash scripts/03_run_ot.sh
GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR PCA_OT_DIR=$PCA_OT_DIR CONFIG=$CONFIG_FILE_REMATCH_DIFF bash scripts/04_train_diffusion_from_dataset.sh

# bash scripts/run_generation.sh