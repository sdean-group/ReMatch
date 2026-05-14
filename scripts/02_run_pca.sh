#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CONFIG="${CONFIG:-config_generate.yaml}"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
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
TIMES=$(python - <<'PY'
import datetime as dt
import json, random

SEED = 12345
N = 120

start = dt.datetime(2021, 1, 1, 0, 0, 0)
end   = dt.datetime(2021,12,31,23,0,0)  # inclusive

total_hours = int((end - start).total_seconds() // 3600) + 1
if N > total_hours:
    raise ValueError(f"N={N} is larger than total_hours={total_hours}")

rng = random.Random(SEED)
idxs = rng.sample(range(total_hours), N)  # 중복 없이
idxs.sort() 

times = [(start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in idxs]
print(json.dumps(times))
PY
)
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
    -m rematch.generate --config-name="${CONFIG}" \
    ++generation.io.reg_ckpt_filename=$REG_PATH \
    ++generation.io.output_filename=$SOURCE_DATASET \
    ++generation.num_ensembles=1 \
    ++generation.inference_mode="regression"\
    "generation.times=${TIMES}"
    # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019.nc\
    # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019_stats.json\
    # ++generation.load_all_times=True\
    # ++generation.times=null\
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    -m rematch.generate --config-name=$CONFIG \
    ++generation.io.reg_ckpt_filename=$REG_PATH \
    ++generation.io.output_filename=$TARGET_DATASET \
    ++generation.num_ensembles=1 \
    ++generation.inference_mode="regression"\
    "generation.times=${TIMES}"
    # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
    # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json
    # ++generation.load_all_times=True\
    # ++generation.times=null\

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
