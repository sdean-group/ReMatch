echo "Running generating ReMatch S"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
REGRESSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_regression/swinir_step_00100000.pt"
DIFFUSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_s/checkpoints_diffusion/EDMPrecondSuperResolution.0.8000000.mdlus"
SAVE_DIR=/data/hrrr_era5_0528/experiment_result/rematch_s
CONFIG="${CONFIG:-config_generate_600_samples.yaml}"
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
-m rematch.generate_swinir --config-name=$CONFIG \
++generation.io.output_filename=${SAVE_DIR}/600_samples.nc \
++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
++generation.num_ensembles=12 \
++generation.inference_mode="all" # all, regression, diffusion