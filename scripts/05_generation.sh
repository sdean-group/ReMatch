# echo "Running generating ReMatch S"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# cd "${REPO_ROOT}"

# export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
# echo "Running generation rematch s "
# REGRESSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_regression/swinir_step_00100000.pt"
# DIFFUSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
# SAVE_DIR=/data/hrrr_era5_0528/experiment_result/rematch_s
# CONFIG="${CONFIG:-config_generate_600_samples.yaml}"
# CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
# -m rematch.generate_swinir --config-name=$CONFIG \
# ++generation.io.output_filename=${SAVE_DIR}/600_samples.nc \
# ++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
# ++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
# ++generation.num_ensembles=12 \
# ++generation.inference_mode="all" # all, regression, diffusion

# echo "Running generation rematch u "
# REGRESSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_regression/UNet.0.8000000.mdlus"
# DIFFUSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
# SAVE_DIR=/data/hrrr_era5_0528/experiment_result/rematch_u
# CONFIG="${CONFIG:-config_generate_600_samples.yaml}"
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
# -m rematch.generate --config-name=$CONFIG \
# ++generation.io.output_filename=${SAVE_DIR}/600_samples.nc \
# ++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
# ++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
# ++generation.num_ensembles=12 \
# ++generation.inference_mode="all" # all, regression, diffusion

# echo "Running generation corrdiff "
# REGRESSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff/checkpoints_regression/UNet.0.8000000.mdlus"
# DIFFUSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
# SAVE_DIR=/data/hrrr_era5_0528/experiment_result/corrdiff
# CONFIG="${CONFIG:-config_generate_600_samples.yaml}"
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
# -m rematch.generate --config-name=$CONFIG \
# ++generation.io.output_filename=${SAVE_DIR}/600_samples.nc \
# ++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
# ++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
# ++generation.num_ensembles=12 \
# ++generation.inference_mode="all" # all, regression, diffusion

# echo "Running generation cfg "
# REGRESSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/cfg/checkpoints_regression/UNet.0.8000000.mdlus"
# DIFFUSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/cfg/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
# SAVE_DIR=/data/hrrr_era5_0528/experiment_result/cfg
# CONFIG="${CONFIG:-config_generate_600_samples.yaml}"
# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
# -m rematch.generate --config-name=$CONFIG \
# ++generation.io.output_filename=${SAVE_DIR}/600_samples.nc \
# ++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
# ++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
# ++generation.num_ensembles=12 \
# ++generation.inference_mode="all" # all, regression, diffusion

#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG="${CONFIG:-config_generate_600_samples.yaml}"
NUM_ENSEMBLES="${NUM_ENSEMBLES:-12}"

# mkdir -p logs/generate

run_generate () {
    local NAME="$1"
    local GPU="$2"
    local REGRESSION_BASELINE="$3"
    local DIFFUSION_BASELINE="$4"
    local SAVE_DIR="$5"

    mkdir -p "${SAVE_DIR}"

    echo "Running generation ${NAME} on GPU ${GPU}" >&2

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        torchrun \
            --standalone \
            --nproc_per_node=1 \
            -m rematch.generate_swinir \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.io.res_ckpt_filename="${DIFFUSION_BASELINE}" \
            ++generation.num_ensembles="${NUM_ENSEMBLES}" \
            ++generation.inference_mode="all"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

# PID_REMATCH_U=$(run_generate \
#     "rematch_u" \
#     "0" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_regression/UNet.0.8000000.mdlus" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/rematch_u")

# PID_CORRDIFF=$(run_generate \
#     "corrdiff_m" \
#     "1" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff_m/checkpoints_regression/UNet.0.3000064.mdlus" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff_m/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/corrdiff_m")

# PID_REMATCH_U=$(run_generate \
#     "rematch_s" \
#     "2" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_regression/swinir_step_00100000.pt" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/rematch_s")
# PID_CFG=$(run_generate \
#     "cfg" \
#     "3" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/cfg/checkpoints_regression/UNet.0.8000000.mdlus" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/cfg/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/data/hrrr_era5_0528/experiment_result/cfg")
# PID_CORRDIFF=$(run_generate \
#     "corrdiff" \
#     "4" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff/checkpoints_regression/UNet.0.8000000.mdlus" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/corrdiff")
# echo "Started jobs:"
# echo "  rematch_u PID=${PID_REMATCH_U}"
# echo "  corrdiff  PID=${PID_CORRDIFF}"
# # echo "  cfg       PID=${PID_CFG}"

# # wait "${PID_REMATCH_U}"
# wait "${PID_CORRDIFF}"
# # wait "${PID_CFG}"

# echo "All generation jobs completed."



run_generate_unet () {
    local NAME="$1"
    local GPU="$2"
    local REGRESSION_BASELINE="$3"
    local SAVE_DIR="$4"

    mkdir -p "${SAVE_DIR}"

    echo "Running generation ${NAME} on GPU ${GPU}" >&2

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        torchrun \
            --standalone \
            --nproc_per_node=1 \
            -m rematch.generate \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.num_ensembles=1 \
            ++generation.inference_mode="regression"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

# PID_UNET=$(run_generate_unet \
#     "unet" \
#     "5" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/corrdiff/checkpoints_regression/UNet.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/unet")


run_generate_convfno () {
    local NAME="$1"
    local GPU="$2"
    local REGRESSION_BASELINE="$3"
    local SAVE_DIR="$4"

    mkdir -p "${SAVE_DIR}"

    echo "Running generation ${NAME} on GPU ${GPU}" >&2

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        torchrun \
            --standalone \
            --nproc_per_node=1 \
            -m rematch.generate_convfno \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.num_ensembles=1 \
            ++generation.inference_mode="regression"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

# PID_CONVFNO=$(run_generate_convfno \
#     "convfno" \
#     "6" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/convfno/checkpoints_regression/UNet.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/convfno")

run_generate_swinir () {
    local NAME="$1"
    local GPU="$2"
    local REGRESSION_BASELINE="$3"
    local SAVE_DIR="$4"

    mkdir -p "${SAVE_DIR}"

    echo "Running generation ${NAME} on GPU ${GPU}" >&2

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        torchrun \
            --standalone \
            --nproc_per_node=1 \
            -m rematch.generate_swinir \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.num_ensembles=1 \
            ++generation.inference_mode="regression"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

# PID_SWINIR=$(run_generate_swinir \
#     "swinir" \
#     "7" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/swinir/checkpoints_regression/swinir_step_00100000.pt" \
#     "/mnt/hrrr_era5_0528/experiment_result/swinir")

run_generate_cdm () {
    local NAME="$1"
    local GPU="$2"
    local DIFFUSION_BASELINE="$3"
    local SAVE_DIR="$4"

    mkdir -p "${SAVE_DIR}"

    echo "Running generation ${NAME} on GPU ${GPU}" >&2

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        torchrun \
            --standalone \
            --nproc_per_node=1 \
            -m rematch.generate \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
            ++generation.io.res_ckpt_filename="${DIFFUSION_BASELINE}" \
            ++generation.num_ensembles="${NUM_ENSEMBLES}" \
            ++generation.inference_mode="diffusion"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

# PID_CDM=$(run_generate_cdm \
#     "cdm" \
#     "5" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/cdm/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/mnt/hrrr_era5_0528/experiment_result/cdm")





# PID_UNET=$!


# echo "Started jobs:"
# echo "  swinir PID=${PID_CONVFNO}"
# # echo "  unet   PID=${PID_UNET}"

# wait "${PID_CONVFNO}"
# # wait "${PID_UNET}"

# echo "Both generation jobs completed."
# CONFIG="${CONFIG:-config_generate_600_samples.yaml}"

run_generate () {
    local NAME="$1"
    local GPU="$2"
    local REGRESSION_BASELINE="$3"
    local DIFFUSION_BASELINE="$4"
    local UQ_BASELINE="$5"
    local UQ_TYPE="$6"
    local SAVE_DIR="$7"

    mkdir -p "${SAVE_DIR}"

    echo "Running generation ${NAME} on GPU ${GPU}" >&2

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        torchrun \
            --standalone \
            --nproc_per_node=1 \
            -m rematch.generate_uq \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.io.res_ckpt_filename="${DIFFUSION_BASELINE}" \
            ++generation.io.bias_ckpt_filename="${UQ_BASELINE}" \
            ++generation.num_ensembles="${NUM_ENSEMBLES}" \
            ++generation.inference_mode="all"\
            ++uq.type="${UQ_TYPE}"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

PID_UQ_RMSE=$(run_generate \
    "uq_rmse" \
    "0" \
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_regression/UNet.0.8000000.mdlus"\
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/uq_rmse/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/uq_rmse/checkpoints_regression/UNet.0.8000000.mdlus" \
    "rmse"\
    "/mnt/hrrr_era5_0528/experiment_result/uq_rmse")

# PID_UQ_QUANTILES=$(run_generate \
#     "uq_quantiles" \
#     "2" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_regression/UNet.0.8000000.mdlus"\
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/uq_quantiles/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/uq_quantiles/checkpoints_regression/UNet.0.3000064.mdlus" \
#     "quantiles"\
#     "/data/hrrr_era5_0528/experiment_result/uq_quantiles_calibration")

# echo "Started jobs:"
# echo "  rematch_u PID=${PID_UQ_RMSE}"
# echo "  corrdiff  PID=${PID_UQ_QUANTILES}"

# wait "${PID_UQ_RMSE}"
# wait "${PID_UQ_QUANTILES}"
# echo "All generation jobs completed."
# run_generate () {
#     local NAME="$1"
#     local GPU="$2"
#     local DIFFUSION_BASELINE="$3"
#     local SAVE_DIR="$4"

#     mkdir -p "${SAVE_DIR}"

#     echo "Running generation ${NAME} on GPU ${GPU}" >&2

#     (
#         export CUDA_VISIBLE_DEVICES="${GPU}"

#         torchrun \
#             --standalone \
#             --nproc_per_node=1 \
#             -m rematch.generate \
#             --config-name="${CONFIG}" \
#             ++generation.io.output_filename="${SAVE_DIR}/600_samples.nc" \
#             ++generation.io.res_ckpt_filename="${DIFFUSION_BASELINE}" \
#             ++generation.num_ensembles="${NUM_ENSEMBLES}" \
#             ++generation.inference_mode="diffusion"
#     ) > "logs/generate/${NAME}.log" 2>&1 &

#     echo $!
# }

# PID_CDM=$(run_generate \
#     "cdm" \
#     "0" \
#     "/home/nvidia/projects/ReMatch/outputs/checkpoints/cdm/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
#     "/data/hrrr_era5_0528/experiment_result/cdm")

# echo "Started jobs:"
# echo "  rematch_u PID=${PID_CDM}"

# wait "${PID_CDM}"
# echo "All generation jobs completed."