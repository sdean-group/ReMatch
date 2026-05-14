#!/bin/bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
# '''times for particle trajectory analysis'''
TIMES=$(python - <<'PY'
import datetime as dt
import json

start = dt.datetime(2021, 6,2, 0, 0, 0)
end   = dt.datetime(2021,6,2,2,0,0)  # inclusive

total_hours = int((end - start).total_seconds() // 3600) + 1

times = [
    (start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")
    for i in range(total_hours)
]

print(json.dumps(times))
PY
)


REGRESSION_BASELINE="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_train/blepressures_temporal/checkpoints_regression/CorrDiffRegressionUNet.0.8000000.mdlus"
DIFFUSION_BASELINE="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_train/blepressures_temporal/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
REGRESSION_TINY_BASELINE="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/regression/tiny_model_2018-2020/checkpoints_regression/CorrDiffRegressionUNet.0.3000064.mdlus"
DIFFUSION_TINY_BASELINE="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/diffusion/diffusion_w_regression_mini_2018-2020/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"

REGRESSION_2018_2019="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/regression/regression_2018-2019/checkpoints_regression/CorrDiffRegressionUNet.0.8000000.mdlus"


DIFFUSION_OT_REG="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/diffusion/diffusion_optimal_transport_top_m_regulation/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
DIFFUSION_OT_TOPM="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/diffusion/diffusion_optimal_transport_knn32_topm3_max_sources_per_target6_alpha1.0_lambda1.0_reg0.1_sinkhorn_iter1500/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
DIFFUSION_OT_REG_200MODES="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/diffusion/diffusion_optimal_transport_top_m_regulation_200modes/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"

RELIABILITY_MAE="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/bias_corrector/2020_rmse_q90/checkpoints_regression/CorrDiffRegressionUNet.0.3000000.mdlus"
DIFFUSION_GT_MAE="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/bias_corrector/diffusion_2020_GT_rmse_regression_2018-2019/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
DIFFUSION_MAE="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/bias_corrector/diffusion_2020_rmse_q90/rmse_only_l2_loss_fn/EDMPrecondSuperResolution.0.8000000.mdlus"

RELIABILITY_Q5Q95="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_uncertainty/verifier_q5q95_val_3milreg/checkpoints_verifier/VerifierUNet.0.8000000.mdlus"
DIFFUSION_Q5="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_uncertainty/verifier_q5q95_val_3milreg/diffusion_masked_l2/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
DIFFUSION_GT_Q5="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_uncertainty/verifier_q5q95_val_3milreg/diffusion_maskedtruth_l2/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"

DIFFUSION_CFG="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_train/blepressures_temporal/class_free_guidance/EDMPrecondSuperResolution.0.8000000.mdlus"
DIFFUSION_CDM="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_train/blepressures_temporal/onlydiffusion/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
DIFFUSION_RDIT="/home/nvidia/projects/corrdiff_original/checkpoints/hrrr_mini_east_train/blepressures_temporal/rdit_fix1/EDMPrecondSuperResolution.0.8000000.mdlus"

SWINIR_2018_2020="/home/nvidia/projects/corrdiff/sr_baselines/swinlr_m/checkpoints/swinir_step_00120000.pt"
SWINIR_2018_2019="/home/nvidia/projects/corrdiff/sr_baselines/swinlr_2018_2019/checkpoints/swinir_step_00100000.pt"
DIFFUSION_SWINIR="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/diffusion/swinlr_ot/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"

CONFIG_600="config_generate_hrrr_mini_east_600_samples.yaml"
CONFIG_12MONTHLY="config_generate_hrrr_mini_east_temporal.yaml"
# SAVE_DIR="/data/shared_experiment/monthly_12samples"
# TIMES='["2021-01-02T23:00:00","2021-02-02T23:00:00","2021-03-02T23:00:00","2021-04-02T23:00:00","2021-05-02T23:00:00","2021-06-02T23:00:00","2021-07-02T23:00:00","2021-08-02T23:00:00","2021-09-02T23:00:00","2021-10-02T23:00:00","2021-11-02T23:00:00","2021-12-02T23:00:00"]'
SAVE_DIR="/data/shared_experiment/particle_traj/20210602_20210603_48H"

EXECUTE="BASELINE" # "BASELINE" "OT_TOPM" "MAE" "GT_MAE" "Q5" "GT_Q5" "SWINIR" "SWINIR_OT"
# for EXECUTE in "CDM" "RDIT"; do

if [ $EXECUTE == "BASELINE" ]; then
  echo "Running baseline corrdiff"
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    -m rematch.generate --config-name=$CONFIG_600 \
    ++generation.io.output_filename=${SAVE_DIR}/test_simple_code.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
    ++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
    ++generation.num_ensembles=12 \
    "generation.times=${TIMES}" 
fi
if [ $EXECUTE == "OT_REG" ]; then
  echo "Running optimal transport (no penalty regularization) + regression 2018-2019"
  CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/ot_reg.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
    ++generation.io.res_ckpt_filename=$DIFFUSION_OT_REG \
    "generation.times=${TIMES}" 
fi
if [ $EXECUTE == "OT_TOPM" ]; then
  echo "Running optimal transport (top m) + regression 2018-2019"
  CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
    generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/ot_topm.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
    ++generation.io.res_ckpt_filename=$DIFFUSION_OT_TOPM\
    "generation.times=${TIMES}" 
fi
if [ $EXECUTE == "MAE" ]; then
  echo "Running bias corrector (MAE only)"
  CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
    generate_bias_corrector.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/mae.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019\
    ++generation.io.bias_ckpt_filename=$RELIABILITY_MAE\
    ++generation.io.res_ckpt_filename=$DIFFUSION_MAE\
    ++generation.hr_mean_conditioning=True\
    ++bias_shape=[16,16]\
    ++dataset.time_flag=True \
    ++generation.rmse_only=True\
    "generation.times=${TIMES}"
fi
if [ $EXECUTE == "GT_MAE" ]; then
  echo "Running bias corrector (GT MAE only)"
  CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    generate_bias_corrector.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/gt_mae.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019\
    ++generation.io.bias_ckpt_filename=$RELIABILITY_MAE\
    ++generation.io.res_ckpt_filename=$DIFFUSION_GT_MAE\
    ++generation.hr_mean_conditioning=True\
    ++bias_shape=[16,16]\
    ++dataset.time_flag=True \
    ++generation.rmse_only=True\
    "generation.times=${TIMES}"
fi
if [ $EXECUTE == "Q5" ]; then
  echo "Running bias corrector (Q5, Q95) + regression 2018-2019"
  cd ../corrdiff_original
  CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    generate_regandverifier_wdiff.py --config-name=config_generate_hrrr_mini_east_verifier_comparisons.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/q5q95.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
    ++generation.io.res_ckpt_filename=$DIFFUSION_Q5 \
    ++generation.io.verifier_ckpt_filename=$RELIABILITY_Q5Q95\
    "generation.times=${TIMES}" 
fi
if [ $EXECUTE == "GT_Q5" ]; then
  echo "Running bias corrector (GT Q5, Q95) + regression 2018-2019"
  cd ../corrdiff_original
  CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    generate_regandverifier_wdiff.py --config-name=config_generate_hrrr_mini_east_verifier_comparisons.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/gt_g5q95.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
    ++generation.io.res_ckpt_filename=$DIFFUSION_GT_Q5\
    ++generation.io.verifier_ckpt_filename=$RELIABILITY_Q5Q95\
    "generation.times=${TIMES}" 
fi
if [ $EXECUTE == "OT_200" ]; then
  echo "Running optimal transport (top m, 200 modes) + regression 2018-2019"
  
  OUT_NC="/data/shared_experiment/optimal_transport/200modes_diffusion_optimal_transport_knn32_topm3_max_sources_per_target6_alpha1.0_lambda1.0_reg0.1_sinkhorn_iter1500/2021_600samples_8milreg2018-202019_8milres2018-2020.nc"
  CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 \
    generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/ot_200.nc \
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
    ++generation.io.res_ckpt_filename=$DIFFUSION_OT_REG_200MODES 
    # "generation.times=${TIMES}" 
fi

if [ $EXECUTE == "CFG" ]; then
  echo "Running corrdiff with CFG cos0->8, null cond, 18 sampler steps"
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    generate.py --config-name=config_generate_hrrr_mini_east_class_free_guidance.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/cfg.nc \
    ++generation.io.res_ckpt_filename=$REGRESSION_BASELINE \
    ++generation.io.res_ckpt_filename=$DIFFUSION_CFG \
    "generation.times=${TIMES}" 
fi

if [ $EXECUTE == "CDM" ]; then
  echo "Running only diffusion corrdiff"
  CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    generate.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/cdm.nc \
    ++generation.inference_mode="diffusion"\
    ++generation.hr_mean_conditioning=False\
    ++generation.io.res_ckpt_filename=$DIFFUSION_CDM \
    "generation.times=${TIMES}" 
fi

if [ $EXECUTE == "RDIT" ]; then
  echo "Running rdit with multgammas, sigma_trn, lambda_eae, coverage optimization"
  CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
    generate_rdit.py --config-name=config_generate_hrrr_mini_east_rdit_diffusion_temporal.yaml \
    ++generation.io.output_filename=${SAVE_DIR}/rdit.nc \
    ++generation.io.res_ckpt_filename=$REGRESSION_BASELINE \
    ++generation.io.res_ckpt_filename=$DIFFUSION_RDIT \
    "generation.times=${TIMES}" 
fi
if [ $EXECUTE == "SWINIR" ]; then
  echo "Running SWINIR baseline"
  echo "TIMES: ${TIMES}"
  # To load specific time indices, use --time_indices "['2021-01-02T23:00:00','2021-02-02T23:00:00']"
  # To load all time indices from dataset, use --time_all
  # To load time indices specified in config file, use --config config_generate_hrrr_mini_east_600_samples.yaml
  # Default is None, priority : --time_indices > --time_all > --config
  CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
    generate_swinlr.py\
    --checkpoint $SWINIR_2018_2020\
    --data_path /data/corrdiff3d/hrrrmini_east_test_ble_temporal.nc\
    --stats_path /data/corrdiff3d/hrrrmini_east_test_ble_temporal_stats.json\
    --multi_step_sampler\
    --save_dir $SAVE_DIR\
    --experiment_name swinir\
    --time_indices "${TIMES}"
fi
if [ $EXECUTE == "SWINIR_OT" ]; then
  echo "Running SWINIR with ReMatch"
  CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1\
    generate_swinlr_ot.py --config-name=config_generate_hrrr_mini_east_600_samples.yaml\
    ++generation.io.swinlr_checkpoint=$SWINIR_2018_2019\
    ++generation.io.output_filename=${SAVE_DIR}/swinir_ot.nc\
    ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019\
    ++generation.io.res_ckpt_filename=$DIFFUSION_SWINIR\
    ++dataset.type=hrrr_mini_raw\
    ++generation.num_ensembles=12\
    "generation.times=${TIMES}" 
fi
# done

# '''baseline corrdiff'''
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
#   generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=${SAVE_DIR}/baseline.nc \
#   ++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
#   ++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
#   "generation.times=${TIMES}" 

# '''optimal transport (no penalty regularization) + regression 2018-2019'''
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
#   generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=${SAVE_DIR}/ot_reg.nc \
#   ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
#   ++generation.io.res_ckpt_filename=$DIFFUSION_OT_REG \
#   "generation.times=${TIMES}" 

# '''optimal transport (top m) + regression 2018-2019'''
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
#   generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=${SAVE_DIR}/ot_topm.nc \
#   ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019 \
#   ++generation.io.res_ckpt_filename=$DIFFUSION_OT_TOPM \
#   "generation.times=${TIMES}" \

# '''bias corrector (MAE only)'''
# CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
#   generate_bias_corrector.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=${SAVE_DIR}/ot_mae.nc \
#   ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019\
#   ++generation.io.bias_ckpt_filename=$RELIABILITY_MAE\
#   ++generation.io.res_ckpt_filename=$DIFFUSION_MAE\
#   ++generation.hr_mean_conditioning=True\
#   ++bias_shape=[16,16]\
#   ++dataset.time_flag=True \
#   ++generation.rmse_only=True\
#   "generation.times=${TIMES}"

# '''bias corrector (GT MAE only)'''
# CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
#   generate_bias_corrector.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=${SAVE_DIR}/ot_gt_mae.nc \
#   ++generation.io.reg_ckpt_filename=$REGRESSION_2018_2019\
#   ++generation.io.bias_ckpt_filename=$RELIABILITY_MAE\
#   ++generation.io.res_ckpt_filename=$DIFFUSION_GT_MAE\
#   ++generation.hr_mean_conditioning=True\
#   ++bias_shape=[16,16]\
#   ++dataset.time_flag=True \
#   ++generation.rmse_only=True\
#   "generation.times=${TIMES}"




  # "generation.times=${TIMES}"\
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29509 --nproc_per_node=4 \
#   train_autoguidance.py --config-name=config_training_hrrr_mini_east_diffusion_temporal.yaml \
#   ++training.io.checkpoint_dir=$SAVE_DIR \
#   ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019.nc\
#   ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019_stats.json \
#   ++training.hp.training_duration=8000000\
#   ++training.io.save_checkpoint_freq=1000000\
#   ++training.io.regression_checkpoint_path=$LATER_REGRESSION\
#   ++training.io.regression_checkpoint_path_2=$EARLIER_REGRESSION\
# CUDA_VISIBLE_DEVICES=2 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29505 --nproc_per_node=1 \
#   generate_autoguidance.py --config-name=config_generate_hrrr_mini_east_class_free_guidance.yaml \
#   ++generation.cfg_weight=3.0\
#   ++generation.io.output_filename=./outputs/autoguidance/earlier_only_50_samples.nc\
#   ++generation.io.res_ckpt_filename=./checkpoints/hrrr_mini_east_train/autoguidance_lpips/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus\
#   ++generation.io.reg_ckpt_filename=$EARLIER_REGRESSION\
#   ++generation.io.reg_ckpt_filename_2=$EARLIER_REGRESSION\
#   ++generation.hr_mean_conditioning=True\
#   "generation.times=${TIMES}"

# Baseline diffusion + regression for 2018-2019
# SAVE_DIR="./checkpoints/hrrr_mini_east_train/diffusion_2020_only"
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29501 --nproc_per_node=4 \
#   train_ble_pressure.py --config-name=config_training_hrrr_mini_east_diffusion_temporal.yaml \
#   ++training.io.checkpoint_dir=$SAVE_DIR \
#   ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
#   ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json \
#   ++training.hp.training_duration=9000000\
#   ++training.io.save_checkpoint_freq=1000000\
#   ++training.io.regression_checkpoint_path=$LATER_REGRESSION


# CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29502 --nproc_per_node=1 \
#   generate_all_reg.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=/data/experiment_outputs/calibration_purpose/regression_mini_target_samples_2020.nc\
#   ++generation.io.reg_ckpt_filename=$LATER_REGRESSION \
#   ++generation.hr_mean_conditioning=True \
#   ++generation.inference_mode=regression \
#   "generation.times=${TIMES}" \
#   ++generation.load_all_times=False \
#   ++generation.num_ensembles=1\
#   ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_ble_temporal.nc \
#   ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_ble_temporal_stats.json
  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json 
  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019.nc\
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019_stats.json 

# EARLIER_REGRESSION="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/regression/regression_2018-2019/checkpoints_regression/CorrDiffRegressionUNet.0.100096.mdlus"
# LATER_REGRESSION="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/regression/regression_2018-2019/checkpoints_regression/CorrDiffRegressionUNet.0.8000000.mdlus"
# REGRESSION_tiny_2018_2019="/home/nvidia/projects/corrdiff/checkpoints/hrrr_mini_east_train/regression/tiny_model_2018-2019/checkpoints_regression/CorrDiffRegressionUNet.0.3000064.mdlus"


# BASELINE_DIFFUSION="./checkpoints/hrrr_mini_east_train/diffusion/diffusion_2020_only/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"

# CUDA_VISIBLE_DEVICES=1 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29507 --nproc_per_node=1 \
#   generate_ble_pressure.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=/data/experiment_outputs/calibration_purpose/regression_2020_3000_samples_trained_on_2018-2019.nc \
#   ++generation.io.reg_ckpt_filename=$LATER_REGRESSION \
#   ++generation.hr_mean_conditioning=True \
#   ++generation.inference_mode=regression \
#   "generation.times=${TIMES}" \
#   ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
#   ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json 

    # ++generation.io.res_ckpt_filename=$BASELINE_DIFFUSION\
  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json 
  
  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019.nc\
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_2018_2019_stats.json 

# FIT PCA TO NEW DATASET COLLECTED WITH TRAINED ON 2018-9 DATASET 
# cd /home/nvidia/projects/calibration
# VALIDATION_NC="/home/nvidia/projects/calibration/dataset/baseline_trainset_2018-2019_120_samples_at_2020.nc"
# WEIGHT_PAIRS="/home/nvidia/projects/calibration/pca/val_pca_weight_pairs_2020yr_120samples.npz"
# # python fit_pca.py \
# #   --validation_nc=$VALIDATION_NC \
# #   --path_weight_pairs=$WEIGHT_PAIRS

# python fit_alpha_film.py \
#   --input=$WEIGHT_PAIRS \
#   --data_nc=$VALIDATION_NC \
#   --load_checkpoint=./pca/sample-wise_alpha_2020yr_120samples.pt \
#   --output=./pca/sample-wise_alpha_trained_on_2020_2021yr_120samples.npz
#sample-wise_alpha_2020yr_120samples.pt why sample-wise_alpha is so good but not others? 

# Bias corrector 
# SAVE_DIR="./checkpoints/hrrr_mini_east_train/bias_corrector/diffusion_2020_GT_rmse"
# BIAS_CORRECTOR_CHECKPOINT="./checkpoints/hrrr_mini_east_train/bias_corrector/2020_rmse_q90/checkpoints_regression/CorrDiffRegressionUNet.0.3000000.mdlus"
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29501 --nproc_per_node=2 \
#   train_bias_corrector.py --config-name=config_training_hrrr_mini_east_diffusion_temporal.yaml \
#   ++training.io.regression_checkpoint_path=$LATER_REGRESSION \
#   ++training.io.bias_checkpoint_path=$BIAS_CORRECTOR_CHECKPOINT \
#   ++training.io.checkpoint_dir=$SAVE_DIR \
#   ++training.hp.training_duration=8000000 \
#   ++training.io.save_checkpoint_freq=1000000 \
#   ++dataset.time_flag=False \
#   ++training.hp.total_batch_size=128


# SAVE_DIR="./checkpoints/hrrr_mini_east_train/bias_corrector/diffusion_2020_GT_rmse_tiny_regression_2018-2019"
# BIAS_CORRECTOR_CHECKPOINT="./checkpoints/hrrr_mini_east_train/bias_corrector/2020_rmse_q90/checkpoints_regression/CorrDiffRegressionUNet.0.3000000.mdlus"
# CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29502 --nproc_per_node=2 \
#   train_bias_corrector.py --config-name=config_training_hrrr_mini_east_diffusion_temporal.yaml \
#   ++training.io.regression_checkpoint_path=$REGRESSION_tiny_2018_2019 \
#   ++training.io.bias_checkpoint_path=$BIAS_CORRECTOR_CHECKPOINT \
#   ++training.io.checkpoint_dir=$SAVE_DIR \
#   ++training.hp.training_duration=8000000 \
#   ++training.io.save_checkpoint_freq=1000000 \
#   ++dataset.time_flag=False \
#   ++training.hp.total_batch_size=128

# SAVE_DIR="./checkpoints/hrrr_mini_east_train/bias_corrector/2020_rmse_q90"
# CUDA_VISIBLE_DEVICES=4,5,6 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29501 --nproc_per_node=3 \
#   train_bias_corrector.py --config-name=config_training_hrrr_mini_east_regression.yaml \
#   ++training.io.regression_checkpoint_path=$LATER_REGRESSION \
#   ++training.io.checkpoint_dir=$SAVE_DIR \
#   ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
#   ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json\
#   ++training.hp.training_duration=3000000 \
#   ++training.io.save_checkpoint_freq=1000000 \
#   ++bias_shape=[16,16]\
#   ++training.hp.total_batch_size=192


  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_ble_temporal.nc \
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_ble_temporal_stats.json\

# SAVE_DIR="./checkpoints/hrrr_mini_east_train/bias_corrector/2020_rmse_q90"                                 
# BIAS_CORRECTOR_CHECKPOINT="${SAVE_DIR}/checkpoints_regression/CorrDiffRegressionUNet.0.3000000.mdlus"
# CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29506 --nproc_per_node=1 \
#   generate_bias_corrector.py --config-name=config_generate_hrrr_mini_east_temporal.yaml \
#   ++generation.io.output_filename=/data/experiment_outputs/hrrr_mini_east/bias_corrector/bias_map/2021_100_samples_rmse_q90_trained_on_2020.nc\
#   ++generation.io.reg_ckpt_filename=$LATER_REGRESSION\
#   ++generation.io.bias_ckpt_filename=$BIAS_CORRECTOR_CHECKPOINT\
#   ++generation.hr_mean_conditioning=True\
#   "generation.times=${TIMES}"\
#   ++bias_shape=[16,16]\
#   ++dataset.time_flag=True \
#   ++generation.num_ensembles=1\
#   ++generation.inference_mode=regression

  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json\
# CHECKPOINT_DIR="./checkpoints/hrrr_mini_east_train/bias_corrector"                                 
# BIAS_CORRECTOR_CHECKPOINT="${CHECKPOINT_DIR}/2020_rmse_q90/checkpoints_regression/CorrDiffRegressionUNet.0.3000000.mdlus"
# RESIDUAL_CHECKPOINT="${CHECKPOINT_DIR}/diffusion_2020_GT_rmse_tiny_regression_2018-2019/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
# OUT_PATH="/data/experiment_outputs/hrrr_mini_east/bias_corrector/2021_10_samples_rmse_trained_on_2020_regression_diffusion_trained_on_2018-2020_l2_loss_fn_GT_rmse.nc"
# # OUT_PATH="/data/shared_experiment/uncertainty_estimator/scalars/rmse/2021_600samples_gt_rmse_3miltinyreg2018-2019_3milbias2020w3milreg_8milres2018-2020_resl2loss.nc"
  

# CONFIG_FILE="config_generate_hrrr_mini_east_temporal.yaml"
# CUDA_VISIBLE_DEVICES=2 torchrun --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29524 --nproc_per_node=1 \
#   generate_bias_corrector.py --config-name=$CONFIG_FILE \
#   ++generation.io.output_filename=$OUT_PATH\
#   ++generation.io.reg_ckpt_filename=$REGRESSION_tiny_2018_2019\
#   ++generation.io.bias_ckpt_filename=$BIAS_CORRECTOR_CHECKPOINT\
#   ++generation.io.res_ckpt_filename=$RESIDUAL_CHECKPOINT\
#   ++generation.hr_mean_conditioning=True\
#   ++bias_shape=[16,16]\
#   ++generation.num_ensembles=12\
#   ++dataset.time_flag=True \
#   ++generation.rmse_only=True\
  # "generation.times=${TIMES}"\
  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_train_ble_temporal.nc \
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_train_ble_temporal_stats.json\
  # ++dataset.data_path=/data/corrdiff3d/hrrrmini_east_validation_2020.nc\
  # ++dataset.stats_path=/data/corrdiff3d/hrrrmini_east_validation_2020_stats.json\