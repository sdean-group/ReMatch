#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_training_rematch_diffusion.yaml}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/checkpoints/rematch_u}"

PCA_OT_DIR="${PCA_OT_DIR:-./outputs/pca_ot}"

echo "Running diffusion from pre-generatedOT dataset"
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
-m rematch.train_preload_diffusion \
--config-name="${CONFIG}" \
++training.io.checkpoint_dir="${CHECKPOINTS_DIR}" \
++dataset.data_path="${PCA_OT_DIR}/reg_2018-2020_ot.nc" 
