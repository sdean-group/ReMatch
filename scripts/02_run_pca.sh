#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_generate.yaml}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./outputs/hrrr_era5/checkpoints/rematch_u}"
SOURCE_DATASET="${SOURCE_DATASET:-./outputs/hrrr_era5/pca_ot/reg_trainset.nc}"
TARGET_DATASET="${TARGET_DATASET:-./outputs/hrrr_era5/pca_ot/reg_calibrationset.nc}"
REG_PATH=$(ls -1 "$CHECKPOINTS_DIR/checkpoints_regression"/CorrDiffRegressionUNet.0.*.mdlus \
  | sort -V \
  | tail -n 1)
PCA_OT_DIR="${PCA_OT_DIR:-./outputs/hrrr_era5/pca_ot}"
GENERATE_DATA="${GENERATE_DATA:-false}"

if [[ ! -f "${SOURCE_DATASET}" || ! -f "${TARGET_DATASET}" ]]; then
  echo "Residual dataset not found. Setting GENERATE_DATA=true"
  echo "SOURCE_DATASET: ${SOURCE_DATASET}"
  echo "TARGET_DATASET: ${TARGET_DATASET}"
  GENERATE_DATA=true
fi

if $GENERATE_DATA; then
  python -m rematch.generate --config-name="${CONFIG}" \
    ++generation.io.reg_ckpt_filename=$REG_PATH \
    ++generation.io.output_filename=$SOURCE_DATASET \
    ++generation.num_ensembles=1 \
    ++generation.inference_mode="regression"\
    ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019.nc\
    ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019_stats.json\
    ++generation.load_all_times=True\
    ++generation.times=null
    # "generation.times=${TIMES}"    
  python -m rematch.generate --config-name=$CONFIG \
    ++generation.io.reg_ckpt_filename=$REG_PATH \
    ++generation.io.output_filename=$TARGET_DATASET \
    ++generation.num_ensembles=1 \
    ++generation.inference_mode="regression"\
    ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
    ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json\
    ++generation.load_all_times=True\
    ++generation.times=null

fi
if $RUN_PCA; then
  echo "Running PCA for $input_variable_name"
  cd ~/projects/ReMatch
  python -m rematch.optimal_transport.fit_pca\
      --training_nc $SOURCE_DATASET\
      --validation_nc $TARGET_DATASET\
      --path_pca_model ${PCA_OT_DIR}/pca_model_residuals.npz\
      --path_weight_pairs ${PCA_OT_DIR}/pca_weights_residuals.npz\
      --n_modes 100\
      --mean_center True
  python -m rematch.optimal_transport.fit_pca\
      --training_nc $SOURCE_DATASET\
      --validation_nc $TARGET_DATASET\
      --path_pca_model ${PCA_OT_DIR}/pca_model_input.npz\
      --path_weight_pairs ${PCA_OT_DIR}/pca_weights_input.npz\
      --n_modes 30\
      --mean_center True\
      --is_lr True
fi
