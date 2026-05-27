echo "Running baseline corrdiff"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
REGRESSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_regression/CorrDiffRegressionUNet.0.8000000.mdlus"
DIFFUSION_BASELINE="/home/nvidia/projects/ReMatch/outputs/checkpoints/rematch_u/checkpoints_diffusion/UNet.0.8000000.mdlus"
CONFIG="${CONFIG:-config_generate.yaml}"
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
-m rematch.generate --config-name=$CONFIG \
++generation.io.output_filename=${SAVE_DIR}/generated_samples.nc \
++generation.io.reg_ckpt_filename=$REGRESSION_BASELINE \
++generation.io.res_ckpt_filename=$DIFFUSION_BASELINE \
++generation.num_ensembles=12 \
++generation.inference_mode="all" # all, regression, diffusion