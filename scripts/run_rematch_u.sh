#!/bin/bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-1}"
OUTPUT_DIR="./outputs/hrrr_era5"
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/rematch_u"
PCA_OT_DIR="${OUTPUT_DIR}/pca_ot"

SOURCE_DATASET=./outputs/hrrr_era5/pca_ot/reg_trainset.nc
TARGET_DATASET=./outputs/hrrr_era5/pca_ot/reg_calibrationset.nc


# CONFIG_FILE_CORRDIFF_REG="config_training_corrdiff_regression.yaml"
# CONFIG_FILE_CORRDIFF_DIFF="config_training_corrdiff_diffusion.yaml"
CONFIG_FILE_REMATCH_REG="config_training_rematch_regression.yaml"
CONFIG_FILE_REMATCH_DIFF="config_training_rematch_diffusion.yaml"
CONFIG_FILE_GEN="config_generate.yaml"

# TRAIN_REG=false
# TRAIN_DIFF=true
# GENERATE_DATA=false
# RUN_PCA=false
# RUN_OT=false
# ''' multiple input levels -> trainset : 405, testset : 51, valset : 50'''
# '''multiple input levels train splits -> trainset : 269, calibrationset : 135'''

# GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_REMATCH_REG bash scripts/01_train_regression.sh
SOURCE_DATASET=$SOURCE_DATASET TARGET_DATASET=$TARGET_DATASET CHECKPOINTS_DIR=$CHECKPOINTS_DIR CONFIG=$CONFIG_FILE_GEN bash scripts/02_run_pca.sh
GPUS=$GPUS NPROC=$NPROC SOURCE_DATASET=$SOURCE_DATASET TARGET_DATASET=$TARGET_DATASET PCA_OT_DIR=$PCA_OT_DIR bash scripts/03_run_ot.sh
GPUS=$GPUS NPROC=$NPROC CHECKPOINTS_DIR=$CHECKPOINTS_DIR PCA_OT_DIR=$PCA_OT_DIR CONFIG=$CONFIG_FILE_REMATCH_DIFF bash scripts/04_train_diffusion.sh
# echo "Running regression"
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone \
#   -m rematch.train --config-name=$CONFIG_FILE_REMATCH_REG \
#   ++training.io.checkpoint_dir=$CHECKPOINTS_DIR\
#   ++training.hp.training_duration=3000\
#   ++training.io.save_checkpoint_freq=3000
# REG_PATH=$(ls -1 "$CHECKPOINTS_DIR/checkpoints_regression"/CorrDiffRegressionUNet.0.*.mdlus \
#   | sort -V \
#   | tail -n 1)

# OUT_FILE_SOURCE=${PCA_OT_DIR}/reg_trainset.nc
# OUT_FILE_TARGET=${PCA_OT_DIR}/reg_calibrationset.nc

# #/data/experiment_outputs/calibration_purpose/regression_tiny_source_samples_2018-2019.nc --validation_nc /data/experiment_outputs/calibration_purpose/regression_tiny_target_samples_2020.nc --path_pca_model ./pca/pca_tiny_model_5.npz --path_weight_pairs /data/experiment_outputs/calibration_purpose/pca_tiny_weights_2018-2019_and_2020.npz --n_modes 5
# #/data/experiment_outputs/calibration_purpose/regression_tiny_source_samples_2018-2019.nc --validation_nc /data/experiment_outputs/calibration_purpose/regression_tiny_target_samples_2020.nc --path_pca_model ./pca/pca_tiny_lr_model_3.npz --path_weight_pairs /data/experiment_outputs/calibration_purpose/pca_tiny_weights_lr_2018-2019_and_2020.npz --n_modes 3 --is_lr True
    

# if $GENERATE_DATA; then
# TIMES=$(python - <<'PY'
# import datetime as dt
# import json, random

# SEED = 12345
# N = 120

# start = dt.datetime(2021, 1, 1, 0, 0, 0)
# end   = dt.datetime(2021,12,31,23,0,0)  # inclusive

# total_hours = int((end - start).total_seconds() // 3600) + 1
# if N > total_hours:
#     raise ValueError(f"N={N} is larger than total_hours={total_hours}")

# rng = random.Random(SEED)
# idxs = rng.sample(range(total_hours), N)  # 중복 없이
# idxs.sort() 

# times = [(start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i in idxs]
# print(json.dumps(times))
# PY
# )
#   CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
#     -m rematch.generate --config-name=$CONFIG_FILE_GEN \
#     ++generation.io.output_filename=$OUT_FILE_SOURCE \
#     ++generation.io.reg_ckpt_filename=$REG_PATH \
#     ++generation.num_ensembles=1 \
#     ++generation.inference_mode="regression"\
#     "generation.times=${TIMES}"
#     # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019.nc\
#     # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019_stats.json\
#     # ++generation.load_all_times=True\
#     # ++generation.times=null\
#   CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
#     -m rematch.generate --config-name=$CONFIG_FILE_GEN \
#     ++generation.io.output_filename=$OUT_FILE_TARGET \
#     ++generation.io.reg_ckpt_filename=$REG_PATH \
#     ++generation.num_ensembles=1 \
#     ++generation.inference_mode="regression"\
#     "generation.times=${TIMES}"
#     # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
#     # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json
#     # ++generation.load_all_times=True\
#     # ++generation.times=null\

# fi
# if $RUN_PCA; then
#   echo "Running PCA for $input_variable_name"
#   cd ~/projects/ReMatch
#   python -m rematch.optimal_transport.fit_pca\
#       --training_nc $OUT_FILE_SOURCE\
#       --validation_nc $OUT_FILE_TARGET\
#       --path_pca_model ${PCA_OT_DIR}/pca_model_residuals.npz\
#       --path_weight_pairs ${PCA_OT_DIR}/pca_weights_residuals.npz\
#       --n_modes 100\
#       --mean_center True
#   python -m rematch.optimal_transport.fit_pca\
#       --training_nc $OUT_FILE_SOURCE\
#       --validation_nc $OUT_FILE_TARGET\
#       --path_pca_model ${PCA_OT_DIR}/pca_model_input.npz\
#       --path_weight_pairs ${PCA_OT_DIR}/pca_weights_input.npz\
#       --n_modes 30\
#       --mean_center True\
#       --is_lr True
# fi

# if $RUN_OT; then
#   echo "Running optimal transport"
#   CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
#   -m rematch.optimal_transport.optimal_transport \
#   --residual_npz_path ${PCA_OT_DIR}/pca_weights_residuals.npz \
#   --condition_npz_path ${PCA_OT_DIR}/pca_weights_input.npz \
#   --path_pca_model ${PCA_OT_DIR}/pca_model_residuals.npz \
#   --source_nc_path $OUT_FILE_SOURCE \
#   --target_nc_path $OUT_FILE_TARGET \
#   --save_dir ${PCA_OT_DIR}
# fi

# if $TRAIN_DIFF; then
#   echo "diffusion training"
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2\
#     -m rematch.train_preload_diffusion \
#     --config-name=$CONFIG_FILE_REMATCH_DIFF \
#     ++training.io.checkpoint_dir=$CHECKPOINTS_DIR \
#     ++training.hp.total_batch_size=16\
#     ++dataset.data_path=${PCA_OT_DIR}/reg_2018-2020_ot.nc\
#     ++training.hp.training_duration=3000\
#     ++training.io.save_checkpoint_freq=3000

#   DIFF_PATH=$(ls -1 "$CHECKPOINTS_DIR/checkpoints_diffusion"/EDMPrecondSuperResolution.0.*.mdlus \
#     | sort -V \
#     | tail -n 1)

#   echo "DIFF_PATH=$DIFF_PATH"

#   TIMES='["2021-06-02T00:00:00","2021-06-02T01:00:00","2021-06-02T02:00:00"]'

#   OUT_FILE=$OUTPUT_DIR/test_rematch_u.nc
#   echo "Running generation for $input_variable_name"
#   CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
#   -m rematch.generate --config-name=$CONFIG_FILE_GEN \
#   ++generation.io.output_filename=$OUT_FILE \
#   ++generation.io.reg_ckpt_filename=$REG_PATH \
#   ++generation.io.res_ckpt_filename=$DIFF_PATH \
#   ++generation.num_ensembles=12 \
#   "generation.times=${TIMES}"
# fi