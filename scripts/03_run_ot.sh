#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

PCA_OT_DIR="${PCA_OT_DIR:-./outputs/hrrr_era5/pca_ot}"
SOURCE_DATASET="${SOURCE_DATASET:-./outputs/hrrr_era5/pca_ot/reg_trainset.nc}"
TARGET_DATASET="${TARGET_DATASET:-./outputs/hrrr_era5/pca_ot/reg_calibrationset.nc}"

echo "Running optimal transport"
python -m rematch.optimal_transport.optimal_transport \
--residual_npz_path ${PCA_OT_DIR}/pca_weights_residuals.npz \
--condition_npz_path ${PCA_OT_DIR}/pca_weights_input.npz \
--path_pca_model ${PCA_OT_DIR}/pca_model_residuals.npz \
--source_nc_path $SOURCE_DATASET \
--target_nc_path $TARGET_DATASET \
--save_dir ${PCA_OT_DIR}