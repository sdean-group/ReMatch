# =====================================================================
# File: helpers/train_bootstrap.py
# =====================================================================
# 공통 초기화/빌드 로직을 train.py / train_bagger.py가 공유하기 위한 모듈
#
# - 박사님 기존 train.py의 대부분을 "함수"로 감싼 형태입니다.
# - 여기서는 "최소한"만 제공합니다.
# - 사용처: train_bagger.py
# =====================================================================

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)
from physicsnemo.metrics.diffusion import ResidualLoss
from physicsnemo.models.diffusion import EDMPrecondSuperResolution

# Student-t (experimental)
from physicsnemo.experimental.metrics.diffusion import tEDMResidualLoss
from physicsnemo.experimental.models.diffusion.preconditioning import tEDMPrecondSuperRes

from datasets.dataset import init_train_valid_datasets_from_config, register_dataset
from physicsnemo.models.diffusion.patching import RandomPatching2D

from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
)

# ---------- same registry objects you already have ----------
# NOTE: 여기서는 "train_bagger.py"가 MODEL_REGISTRY를 자체 포함한다고 가정합니다.
#       만약 박사님이 registry도 공유하고 싶으시면 이 파일로 옮겨도 됩니다.


def resolve_diffusion_backend(cfg: DictConfig, logger0):
    distribution = getattr(cfg.training.hp, "distribution", None)
    student_t_nu = getattr(cfg.training.hp, "student_t_nu", None)

    residual_loss_cls = ResidualLoss
    precond_cls = EDMPrecondSuperResolution

    if distribution not in ["normal", "student_t", None]:
        raise ValueError(f"Invalid distribution {distribution}")

    if distribution == "student_t":
        if student_t_nu is None:
            raise ValueError("student_t_nu must be provided in cfg.training.hp.student_t_nu for student_t distribution")
        if student_t_nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {student_t_nu}.")
        residual_loss_cls = tEDMResidualLoss
        precond_cls = tEDMPrecondSuperRes
        logger0.info(
            f"Using student-t distribution with nu={student_t_nu}. "
            f"This is an experimental feature and APIs may change without notice."
        )
    return residual_loss_cls, precond_cls, distribution, student_t_nu


def setup_dist_and_loggers(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    initialize_wandb(
        project="Modulus-Launch",
        entity="yujin1004",
        name=f"CorrDiff-Training-{os.getenv('HYDRA_JOB_NAME', 'job')}",
        group="CorrDiff-DDP-Group",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        results_dir=cfg.wandb.results_dir,
    )

    return dist, logger0


def setup_checkpoint_dir(cfg: DictConfig) -> str:
    if cfg.training.io.checkpoint_dir is not None:
        return cfg.training.io.checkpoint_dir
    return get_checkpoint_dir(str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name)


def load_or_init_cur_nimg(checkpoint_dir: str) -> int:
    try:
        return load_checkpoint(path=checkpoint_dir)
    except Exception:
        return 0


def init_datasets(
    cfg: DictConfig,
    cur_nimg: int,
    dist,
):
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None,
    }

    validation = hasattr(cfg, "validation")
    validation_dataset_cfg = OmegaConf.to_container(cfg.validation, resolve=True) if validation else None

    register_dataset(cfg.dataset.type)

    dataset, dataset_iterator, validation_dataset, validation_dataset_iterator = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        validation=validation,
        sampler_start_idx=cur_nimg,
    )
    return dataset, dataset_iterator, validation_dataset, validation_dataset_iterator


def setup_batch_size(cfg: DictConfig, dist):
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = cfg.training.hp.total_batch_size // dist.world_size


def setup_seed_and_cuda(cur_nimg: int, dist):
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()


def setup_patching(cfg: DictConfig, spec, dataset, img_shape, img_in_channels):
    # patching only if spec.is_patched
    if spec.is_patched:
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None

    if (
        patch_shape_x
        and patch_shape_y
        and patch_shape_y >= img_shape[0]
        and patch_shape_x >= img_shape[1]
    ):
        patch_shape_x = None
        patch_shape_y = None

    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape2, patch_shape2 = set_patch_shape(img_shape, patch_shape)

    if spec.forbids_patching and use_patching:
        raise ValueError(f"Model ({cfg.model.name}) forbids patch-based training, but patching is enabled by config.")

    if use_patching:
        patching = RandomPatching2D(
            img_shape=img_shape2,
            patch_shape=patch_shape2,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        # interpolate global channel if patch-based model is used
        img_in_channels += len(dataset.input_channels())
    else:
        patching = None

    return patching, use_patching, img_shape2, img_in_channels


def load_regression_net_if_any(cfg: DictConfig, dist, use_apex_gn: bool, enable_amp: bool, profile_mode: bool):
    if hasattr(cfg.training.io, "regression_checkpoint_path") and cfg.training.io.regression_checkpoint_path is not None:
        path = to_absolute_path(cfg.training.io.regression_checkpoint_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected regression checkpoint but not found: {path}")

        regression_net = Module.from_checkpoint(path, override_args={"use_apex_gn": use_apex_gn})
        regression_net.amp_mode = enable_amp
        regression_net.profile_mode = profile_mode
        regression_net.eval().requires_grad_(False).to(dist.device)
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        return regression_net
    return None


def build_optimizer(cfg: DictConfig, model):
    return torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True,
    )


def compute_grad_acc(cfg: DictConfig, dist):
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    return batch_gpu_total, num_accumulation_rounds


def maybe_load_optimizer(checkpoint_dir: str, optimizer, dist):
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        load_checkpoint(path=checkpoint_dir, optimizer=optimizer, device=dist.device)
    except Exception:
        pass


def maybe_load_model(checkpoint_dir: str, model):
    try:
        load_checkpoint(path=checkpoint_dir, models=model)
    except Exception:
        pass


def save_ckpt(checkpoint_dir: str, model, optimizer, cur_nimg: int):
    save_checkpoint(
        path=checkpoint_dir,
        models=model,
        optimizer=optimizer,
        epoch=cur_nimg,
    )