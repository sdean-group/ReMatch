#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG="${CONFIG:-config_generate.yaml}"
NUM_ENSEMBLES="${NUM_ENSEMBLES:-12}"

mkdir -p logs/generate

# Run Rematch generation 
run_generate_rematchs () {
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
            ++generation.io.output_filename="${SAVE_DIR}/samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.io.res_ckpt_filename="${DIFFUSION_BASELINE}" \
            ++generation.num_ensembles="${NUM_ENSEMBLES}" \
            ++generation.inference_mode="all"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}
run_generate_rematchu () {
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
            -m rematch.generate \
            --config-name="${CONFIG}" \
            ++generation.io.output_filename="${SAVE_DIR}/samples.nc" \
            ++generation.io.reg_ckpt_filename="${REGRESSION_BASELINE}" \
            ++generation.io.res_ckpt_filename="${DIFFUSION_BASELINE}" \
            ++generation.num_ensembles="${NUM_ENSEMBLES}" \
            ++generation.inference_mode="all"
    ) > "logs/generate/${NAME}.log" 2>&1 &

    echo $!
}

PID_REMATCH_S=$(run_generate_rematchs \
    "rematch_s" \
    "0" \
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_regression/swinir_step_00100000.pt" \
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
    "/mnt/hrrr_era5_0528/experiment_result/rematch_s")

PID_REMATCH_U=$(run_generate_rematchu \
    "rematch_u" \
    "1" \
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_regression/UNet.0.8000000.mdlus" \
    "/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus" \
    "/mnt/hrrr_era5_0528/experiment_result/rematch_u_again")

echo "Started jobs:"
echo "  rematch_u PID=${PID_REMATCH_U}"
echo "  rematch_s PID=${PID_REMATCH_S}"

wait "${PID_REMATCH_U}"
wait "${PID_REMATCH_S}"
echo "All generation jobs completed."