# ReMatch
Official implementation of "Mind the Residual Gap: Probabilistic Downscaling under Real-World Bias"
by [Yujin Kim](https://yujin1007.github.io/), [Nidhi Soma](https://github.com/NIDHI2023) and [Sarah Dean &dagger;](https://sdean-group.github.io/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05294-df2a2a.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/pdf/2511.20612)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badges)](LICENSE)
[![Website](https://img.shields.io/badge/🔗-WebSite-black?style=for-the-badge)](https://sdean-group.github.io/ReMatch/)

**ReMatch** (**Re**sidual Target **Match**ing) mitigates residual target misspecification in mean–residual probabilistic downscaling. In real-world downscaling, systematic bias between low-resolution inputs and high-resolution targets can make the residuals seen during training differ from the correction residuals required at test time, causing biased and under-dispersive ensembles.

ReMatch uses optimal transport to align training residual targets with a test-like calibration regime in a low-dimensional PCA space. The pipeline first trains a deterministic mean predictor, then transports the training residual targets toward calibration residuals, and finally trains a conditional diffusion model on the transported residuals. This produces high-resolution ensemble predictions with improved calibration and probabilistic spread.

This repository includes ReMatch with UNet (ReMatch_u), ReMatch with SwinIr (ReMatch_s), Corrdiff and SwinIR implementation on lower starotosphere pressure level ERA5--HRRR wind field downscaling task on northeast CONUS region. 



## 🛠 Environment Setup
This project was tested with Python 3.12.3
```bash
pip install -r requirements.txt
```
### Dataset Generation

You can download and generate the required datasets from [HRRR-ERA5-Dataset](https://github.com/NIDHI2023/HRRR-ERA5-Dataset.git).

ReMatch requires three separate datasets. In our implementation, we used:

- **2018–2019** for training
- **2020** for calibration
- **2021** for testing

After generating the datasets and statistics files, update `dataset_path` and `stat_path` in the configuration files under:

```bash
ReMatch/conf
```

To run ReMatch, you also need to specify the training and calibration dataset paths in:

```bash
ReMatch/scripts/02_run_pca.sh
```

## Training 
### Configuration basics

CorrDiff training is managed through `train.py` and uses YAML configuration files powered by [Hydra](https://hydra.cc/docs/intro/). The configuration system is organized as follows:

- **Run Scripts**: Located in the `scripts` directory. Run script from project root. To begin training, you can specify the number of gpus, for example, 
``
GPUS=0,1,2,3 NPROC=4 bash scripts/run_rematchs.sh
``
  - **Run all at once**:
    - ReMatch_u: 
      - `run_rematchu.sh` - First, train UNet regression model with train dataset. Second, collect residaul target from trained regression model and GT on train and calibration set individually, then run PCA on residual targets and low resolultion input. Third, transport trainset residual target into calibrationset resdiaul target. Lastly, train diffusion model with matched residual target on both train and calibration datset.  
    - ReMatch_s:
      -`run_rematchs.sh` - Same procedure with ReMatch_u but SwinIR regression model
    -Corrdiff:
      -`run_corrdiff.sh` - First train UNet regression model with train+calibration and then train diffusion model with train + calibration dataset 
    -SwinIR:
      -`run_swinir.sh` - Train SwinIR regression model using train+calibration dataset 
  - **Run individual modules**:
    - `01_train_regression_unet.sh / 01_train_regression_swinir.sh` - train regression model. takes configuration file name as an input argument
    - `02_run_pca.sh` - Run PCA on residual target and low resolution input. when there is no source(trainset) and target(calibration) generation nc file exists, run generation before PCA. Takes configuration file name for generation, source and target generation file name and regression type(unet or swinir) as input arguments. 
    -`03_run_ot.sh` - Run optimal transport on PCA latent space of residual targets. Takes PCA results directory, source and target generation file name as input arguments. 
    -`04_train_diffusion_from_regression.sh / 04_train_diffusion_from_dataset` - Train diffusion model using either regression model or pre-generated dataset (matched residual target). 
  - **Run generation**:
    - `05_traingeneration.sh` - submits generation jobs for `rematch_u` and `rematch_s` on separate GPUs. Make sure to update the checkpoint paths to match your trained checkpoints.
    

- **Base Configurations**: Located in the `conf/base` directory
- **Configuration Files**:
  - **Training Configurations**:
    - ReMatch_u: 
      - `conf/config_training_rematchu_regression` - Configuration for training the UNet regression model, trained with training subset. 
      - `conf/config_training_rematch_diffusion` - Configuration for training the diffusion model with pre-calculated matched residual target, trained with full training set (transported train + calibration dataset)
    - ReMatch_s:
      - `conf/config_training_rematchs_regression` - Configuration for training the SwinIR regression model, trained with training subset. 
      - `conf/config_training_rematch_diffusion` - Configuration for training the diffusion model with pre-calculated matched residual target, trained with full training set (transported train + calibration dataset)
    - Corrdiff:
      - `conf/config_training_corrdiff_regression.yaml` - Configuration for training the UNet regression model, trained with full training set (train + calibration dataset)
      - `conf/config_training_corrdiff_diffusion.yaml` - Configuration for training the diffusion model, trained with full training set (train + calibration dataset)
    - SwinIr:
      - `conf/config_training_swinirregression.yaml` - Configuration for training the SwinIR regression model, trained with full training set (train + calibration dataset)
  - **Generation Configurations**:
    - `conf/config_generate.yaml` - Configuration for generating monthly predictions on test dataset. 

**Training Details:**
- Duration: few hours on a 8 A100 GPU
- Checkpointing: Automatically resumes from latest checkpoint if interrupted
- Multi-GPU Support: Compatible with `torchrun` or MPI for distributed training

> **💡 Memory Management**  
> The default configuration uses a batch size of 256 (controlled by `training.hp.total_batch_size`). If you encounter memory constraints, particularly on GPUs with limited memory, you can reduce the per-GPU batch size by setting `++training.hp.batch_size_per_gpu=64` on individual module training.
# CorrDiff baseline
Upstream:
- Repository: NVIDIA/physicsnemo
- Source path: examples/weather/corrdiff
- Package used for this release: nvidia-physicsnemo==1.3.0
- License: Apache-2.0
- Note: This code is adapted from the CorrDiff example in NVIDIA PhysicsNeMo. Some import paths were updated for compatibility with the public PyPI release nvidia-physicsnemo==1.3.0.

# SwinIR baseline

We use a SwinIR-style deterministic super-resolution model as a mean predictor and deterministic baseline.

Upstream:
- Repository: JingyunLiang/SwinIR
- Paper: SwinIR, ICCV 2021
- License: Apache-2.0

Modifications used in this paper:
- Adapted input/output channels for multi-channel wind-field super-resolution.
- Set window size to 7 for 21x21 low-resolution inputs.
- Used PixelShuffle upsampling to map 21x21 inputs to 168x168 outputs.
