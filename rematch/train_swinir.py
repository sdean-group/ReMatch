# train_rematch_swinir.py
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from collections import deque
from contextlib import nullcontext

import hydra
import nvtx
import torch
import torch.nn as nn
import torch.distributed as torch_dist
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from third_party.swinir.swinir import SwinIR2DSR
from rematch.rematch_loss import ResidualLoss_swinir
from third_party.datasets.hrrrmini_raw_lr import HRRRMiniRawLRDataset

from physicsnemo import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.diffusion import EDMPrecondSuperResolution
# from physicsnemo.metrics.diffusion import ResidualLoss
from physicsnemo.utils import load_checkpoint, save_checkpoint, get_checkpoint_dir
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from third_party.datasets.dataset import init_train_valid_datasets_from_config, register_dataset
from third_party.helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)


def cuda_profiler():
    return torch.cuda.profiler.profile() if torch.cuda.is_available() else nullcontext()


def profiler_emit_nvtx():
    return torch.autograd.profiler.emit_nvtx() if torch.cuda.is_available() else nullcontext()



def _has(cfg, key: str) -> bool:
    return isinstance(cfg, DictConfig) and key in cfg


def _get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg[key] if key in cfg else default
    return getattr(cfg, key, default)


def cfg_to_swinir_args(cfg: DictConfig) -> SimpleNamespace:
    m = cfg.swinir.model_args
    t = _get(cfg.swinir, "train", None)

    training_cfg = _get(cfg, "training", None)
    training_io = _get(training_cfg, "io", None)
    training_hp = _get(training_cfg, "hp", None)

    validation_cfg = _get(cfg, "validation", None)

    return SimpleNamespace(
        # ------------------------------------------------------------
        # Model args: required for both train and generate
        # ------------------------------------------------------------
        in_chans=int(m.in_chans),
        out_chans=int(m.out_chans),
        img_size=list(m.img_size),
        upscale=int(m.upscale),
        window_size=int(m.window_size),
        embed_dim=int(m.embed_dim),
        depths=list(m.depths),
        num_heads=list(m.num_heads),
        mlp_ratio=float(m.mlp_ratio),
        drop_path_rate=float(m.drop_path_rate),
        use_checkpoint=bool(m.use_checkpoint),
        multi_step_sampler=bool(m.multi_step_sampler),

        # ------------------------------------------------------------
        # SwinIR train args: optional for generate
        # ------------------------------------------------------------
        loss=str(_get(t, "loss", "l1")),
        lr=float(_get(t, "lr", 1e-4)),
        weight_decay=float(_get(t, "weight_decay", 0.0)),
        grad_clip=float(_get(t, "grad_clip", 0.0)),
        amp=bool(_get(t, "amp", False)),
        num_workers=int(_get(t, "num_workers", 4)),
        val_interval=int(_get(t, "val_interval", 0)),
        val_batch_size=int(_get(t, "val_batch_size", 1)),
        val_max_batches=int(_get(t, "val_max_batches", 0)),

        # ------------------------------------------------------------
        # Dataset args: required for both train and generate
        # ------------------------------------------------------------
        data_path=str(cfg.dataset.data_path),
        stats_path=str(cfg.dataset.stats_path),
        val_data_path=str(_get(validation_cfg, "data_path", cfg.dataset.data_path)),
        val_stats_path=str(_get(validation_cfg, "stats_path", cfg.dataset.stats_path)),

        # ------------------------------------------------------------
        # CorrDiff-style training args: optional for generate
        # ------------------------------------------------------------
        checkpoint_interval=int(_get(training_io, "save_checkpoint_freq", 0)),
        log_interval=int(_get(training_io, "print_progress_freq", 0)),
        total_steps=int(_get(training_hp, "training_duration", 0)),
        batch_size=int(_get(training_hp, "batch_size_per_gpu", 0)),

        # ------------------------------------------------------------
        # Resume: optional
        # ------------------------------------------------------------
        resume=None
        if _get(cfg.swinir, "resume", None) is None
        else str(_get(cfg.swinir, "resume")),
    )
def build_swinir_from_cfg(cfg: DictConfig) -> nn.Module:
    args = cfg_to_swinir_args(cfg)
    return SwinIR2DSR(
        img_size=tuple(args.img_size),
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        upscale=args.upscale,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
        resi_connection="1conv",
        multi_step_sampler=args.multi_step_sampler,
    )


def load_swinir_checkpoint(path: str, model: nn.Module, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[SwinIR checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[SwinIR checkpoint] unexpected keys: {len(unexpected)}")

    return int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0


def save_swinir_checkpoint(path, model, optimizer, scaler, step, cfg, distributed: bool):
    if distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def build_raw_dataset(cfg: DictConfig, split: str):
    if split == "train":
        data_path = cfg.dataset.data_path
        stats_path = cfg.dataset.stats_path
    elif split == "val":
        if hasattr(cfg, "validation"):
            data_path = cfg.validation.data_path
            stats_path = cfg.validation.stats_path
        else:
            data_path = cfg.dataset.data_path
            stats_path = cfg.dataset.stats_path
    else:
        raise ValueError(f"Unknown split: {split}")

    return HRRRMiniRawLRDataset(
        data_path=data_path,
        stats_path=stats_path,
        input_variables=list(cfg.dataset.input_variables),
        output_variables=list(cfg.dataset.output_variables),
    )


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def get_raw_batch(batch):
    # HRRRMiniRawLRDataset returns: hr, lr_raw, lr
    hr, lr_raw, lr = batch
    return hr, lr_raw, lr


@torch.no_grad()
def validate_swinir(model, loader, criterion, device, amp: bool, world_size: int, max_batches: int):
    model.eval()

    loss_sum = torch.zeros((), device=device)
    mae_sum = torch.zeros((), device=device)
    mse_sum = torch.zeros((), device=device)
    count = torch.zeros((), device=device)

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        batch = move_to_device(batch, device)
        hr_target, lr_input, _ = get_raw_batch(batch)

        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(lr_input)
            loss = criterion(pred, hr_target)

        diff = pred.float() - hr_target.float()
        bsz = hr_target.shape[0]
        bcount = torch.tensor(float(bsz), device=device)

        loss_sum += loss.detach().float() * bcount
        mae_sum += diff.abs().mean().detach() * bcount
        mse_sum += diff.pow(2).mean().detach() * bcount
        count += bcount

    if world_size > 1:
        torch_dist.all_reduce(loss_sum, op=torch_dist.ReduceOp.SUM)
        torch_dist.all_reduce(mae_sum, op=torch_dist.ReduceOp.SUM)
        torch_dist.all_reduce(mse_sum, op=torch_dist.ReduceOp.SUM)
        torch_dist.all_reduce(count, op=torch_dist.ReduceOp.SUM)

    model.train()

    return {
        "loss": float((loss_sum / count.clamp_min(1.0)).item()),
        "mae": float((mae_sum / count.clamp_min(1.0)).item()),
        "rmse": float(torch.sqrt(mse_sum / count.clamp_min(1.0)).item()),
    }


def train_swinir_regression(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("swinir_regression")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    OmegaConf.resolve(cfg)
    args = cfg_to_swinir_args(cfg)

    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = cfg.training.hp.total_batch_size // dist.world_size
        args.batch_size = int(cfg.training.hp.batch_size_per_gpu)

    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    checkpoint_root = Path(cfg.training.io.checkpoint_dir)
    ckpt_dir = checkpoint_root / "checkpoints_regression"
    tb_dir = Path("tensorboard") / HydraConfig.get().job.name

    if dist.rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:
        writer = None

    train_dataset = build_raw_dataset(cfg, split="train")
    val_dataset = build_raw_dataset(cfg, split="val")

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        shuffle=True,
        drop_last=True,
    ) if dist.world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        shuffle=False,
        drop_last=False,
    ) if dist.world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = build_swinir_from_cfg(cfg).to(dist.device)

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            find_unused_parameters=False,
        )

    if args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "l2":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported SwinIR loss: {args.loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    step = 0
    if args.resume is not None:
        resume_path = to_absolute_path(args.resume)
        ckpt = torch.load(resume_path, map_location=dist.device)
        state = ckpt["model"]
        if dist.world_size > 1:
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt.get("step", 0))
        logger0.info(f"Resumed SwinIR from {resume_path} at step {step}")

    model.train()
    loss_window = deque(maxlen=max(1, args.log_interval))
    start_time = time.time()

    while step < args.total_steps:
        if sampler is not None:
            sampler.set_epoch(step)

        for batch in train_loader:
            if step >= args.total_steps:
                break

            step += 1
            batch = move_to_device(batch, dist.device)
            hr_target, lr_input, _ = get_raw_batch(batch)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(lr_input)

                if pred.shape != hr_target.shape:
                    raise ValueError(
                        f"SwinIR prediction/target shape mismatch: "
                        f"pred={tuple(pred.shape)}, target={tuple(hr_target.shape)}"
                    )

                loss = criterion(pred, hr_target)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            loss_reduced = loss.detach()
            if dist.world_size > 1:
                torch_dist.all_reduce(loss_reduced, op=torch_dist.ReduceOp.SUM)
                loss_reduced /= dist.world_size

            loss_value = float(loss_reduced.item())
            loss_window.append(loss_value)

            if dist.rank == 0 and step % args.log_interval == 0:
                loss_mean = sum(loss_window) / len(loss_window)
                elapsed = time.time() - start_time
                steps_per_sec = step / max(elapsed, 1e-8)

                writer.add_scalar("train/loss", loss_value, step)
                writer.add_scalar("train/loss_mean", loss_mean, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, step)

                logger0.info(
                    f"[SwinIR step {step}] "
                    f"loss={loss_value:.6e}, "
                    f"loss_mean={loss_mean:.6e}, "
                    f"steps/s={steps_per_sec:.3f}"
                )

            if dist.rank == 0 and step % args.checkpoint_interval == 0:
                ckpt_path = ckpt_dir / f"swinir_step_{step:08d}.pt"
                save_swinir_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    cfg=cfg,
                    distributed=dist.world_size > 1,
                )
                logger0.info(f"Saved SwinIR checkpoint: {ckpt_path}")

            if args.val_interval > 0 and step % args.val_interval == 0:
                metrics = validate_swinir(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=dist.device,
                    amp=args.amp,
                    world_size=dist.world_size,
                    max_batches=args.val_max_batches,
                )

                if dist.rank == 0:
                    writer.add_scalar("val/loss", metrics["loss"], step)
                    writer.add_scalar("val/mae", metrics["mae"], step)
                    writer.add_scalar("val/rmse", metrics["rmse"], step)
                    logger0.info(
                        f"[SwinIR val step {step}] "
                        f"loss={metrics['loss']:.6e}, "
                        f"mae={metrics['mae']:.6e}, "
                        f"rmse={metrics['rmse']:.6e}"
                    )

    if dist.rank == 0:
        final_path = ckpt_dir / f"swinir_final_step_{step:08d}.pt"
        save_swinir_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            cfg=cfg,
            distributed=dist.world_size > 1,
        )
        writer.close()
        logger0.info(f"Saved final SwinIR checkpoint: {final_path}")


def build_frozen_regression_net_for_diffusion(cfg: DictConfig, device: torch.device, logger0):
    if cfg.swinir.checkpoint_path is None:
        raise ValueError(
            "For model.name=diffusion with SwinIR mean prior, "
            "cfg.swinir.checkpoint_path must be provided."
        )

    ckpt_path = to_absolute_path(str(cfg.swinir.checkpoint_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SwinIR checkpoint not found: {ckpt_path}")

    regression_net = build_swinir_from_cfg(cfg)
    step = load_swinir_checkpoint(ckpt_path, regression_net, device)
    regression_net.eval().requires_grad_(False).to(device)

    logger0.success(f"Loaded frozen SwinIR regression model from {ckpt_path} at step {step}")
    return regression_net


def train_diffusion_with_swinir(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.rank == 0:
        writer = SummaryWriter(log_dir=f"tensorboard/{HydraConfig.get().job.name}")
    else:
        writer = None

    logger = PythonLogger("diffusion_swinir")
    logger0 = RankZeroLoggingWrapper(logger, dist)

    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    validation = hasattr(cfg, "validation")
    validation_dataset_cfg = OmegaConf.to_container(cfg.validation) if validation else None

    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if fp_optimizations == "amp-fp16" else torch.bfloat16

    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")),
        cfg.model.name,
    )

    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    try:
        cur_nimg = load_checkpoint(path=checkpoint_dir)
    except Exception:
        cur_nimg = 0

    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()

    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None,
    }

    dataset, dataset_iterator, validation_dataset, validation_dataset_iterator = (
        init_train_valid_datasets_from_config(
            dataset_cfg,
            data_loader_kwargs,
            batch_size=cfg.training.hp.batch_size_per_gpu,
            seed=0,
            validation_dataset_cfg=validation_dataset_cfg,
            validation=validation,
            sampler_start_idx=cur_nimg,
        )
    )

    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }

    if hasattr(cfg.model, "model_args"):
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    if enable_amp:
        model_args["amp_mode"] = enable_amp

    if cfg.model.name != "diffusion":
        raise ValueError(
            f"train_diffusion_with_swinir only supports cfg.model.name='diffusion', "
            f"got {cfg.model.name}"
        )

    model = EDMPrecondSuperResolution(
        img_in_channels=img_in_channels + model_args["N_grid_channels"],
        **model_args,
    )

    model.train().requires_grad_(True).to(dist.device)

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    try:
        load_checkpoint(path=checkpoint_dir, models=model)
    except Exception:
        pass

    regression_net = build_frozen_regression_net_for_diffusion(
        cfg=cfg,
        device=dist.device,
        logger0=logger0,
    )

    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )

    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    P_mean = getattr(cfg.training.hp, "P_mean", None)
    P_std = getattr(cfg.training.hp, "P_std", None)
    sigma_data = getattr(cfg.training.hp, "sigma_data", None)

    loss_init_kwargs = {}
    if P_mean is not None:
        loss_init_kwargs["P_mean"] = P_mean
    if P_std is not None:
        loss_init_kwargs["P_std"] = P_std
    if sigma_data is not None:
        loss_init_kwargs["sigma_data"] = sigma_data

    loss_fn = ResidualLoss_swinir(
        regression_net=regression_net,
        hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        **loss_init_kwargs,
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True,
    )

    if dist.world_size > 1:
        torch.distributed.barrier()

    try:
        load_checkpoint(path=checkpoint_dir, optimizer=optimizer, device=dist.device)
    except Exception:
        pass

    logger0.info(f"Training diffusion for {cfg.training.hp.training_duration} images...")

    done = False
    start_nimg = cur_nimg
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1

    input_dtype = torch.float32
    if fp16 and not enable_amp:
        input_dtype = torch.float16

    start_time = time.time()

    with cuda_profiler():
        with profiler_emit_nvtx():
            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                optimizer.zero_grad(set_to_none=True)
                loss_accum = 0.0

                for n_i in range(num_accumulation_rounds):
                    img_clean, img_lr_raw, img_lr, *lead_time_label  = next(dataset_iterator)

                    img_clean = img_clean.to(dist.device).to(input_dtype).contiguous()
                    img_lr = img_lr.to(dist.device).to(input_dtype).contiguous()
                    img_lr_raw = img_lr_raw.to(dist.device).to(input_dtype).contiguous()

                    loss_fn_kwargs = {
                        "net": model,
                        "img_clean": img_clean,
                        "img_lr": img_lr,
                        "img_lr_raw": img_lr_raw,
                        "augment_pipe": None,
                    }

                    with torch.cuda.amp.autocast(enabled=enable_amp, dtype=amp_dtype):
                        loss = loss_fn(**loss_fn_kwargs)
                    loss = loss.sum() / cfg.training.hp.batch_size_per_gpu
                    loss = loss / num_accumulation_rounds
                    loss_accum += loss.detach()

                    loss.backward()
                grad_clip_threshold = getattr(cfg.training.hp, "grad_clip_threshold", 0.0)
                if grad_clip_threshold is None:
                    grad_clip_threshold = 0.0

                handle_and_clip_gradients(model, float(grad_clip_threshold))
                optimizer.step()

                cur_nimg += batch_gpu_total

                if dist.world_size > 1:
                    torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.SUM)
                    loss_accum /= dist.world_size

                loss_value = float(loss_accum.item())
                average_loss_running_mean += loss_value
                n_average_loss_running_mean += 1

                if is_time_for_periodic_task(
                        cur_nimg,
                        cfg.training.io.print_progress_freq,
                        done,
                        cfg.training.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    ):
                    sec_per_tick = time.time() - tick_start_time
                    logger0.info(
                        f"kimg={cur_nimg / 1000.0:.1f}, "
                        f"loss={loss_value:.6e}, "
                        f"avg_loss={average_loss_running_mean / n_average_loss_running_mean:.6e}, "
                        f"sec/tick={sec_per_tick:.2f}"
                    )

                    if dist.rank == 0 and writer is not None:
                        writer.add_scalar("train/loss", loss_value, cur_nimg)
                        writer.add_scalar(
                            "train/loss_running_mean",
                            average_loss_running_mean / n_average_loss_running_mean,
                            cur_nimg,
                        )

                    average_loss_running_mean = 0
                    n_average_loss_running_mean = 1

                if is_time_for_periodic_task(
                        cur_nimg,
                        cfg.training.io.save_checkpoint_freq,
                        done,
                        cfg.training.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    ):
                    if dist.rank == 0:
                        save_checkpoint(
                            path=checkpoint_dir,
                            models=model,
                            optimizer=optimizer,
                            epoch=cur_nimg,
                        )
                        logger0.info(f"Saved diffusion checkpoint at nimg={cur_nimg}")

                done = cur_nimg >= cfg.training.hp.training_duration

    print(f"Training completed at nimg={cur_nimg}, done={done}")
    if dist.rank == 0 and writer is not None:
        writer.close()


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    if cfg.model.name == "swinir_regression":
        train_swinir_regression(cfg)
    elif cfg.model.name == "diffusion":
        train_diffusion_with_swinir(cfg)
    else:
        raise ValueError(
            "Unsupported model.name for train_rematch_swinir.py. "
            "Use model.name='swinir_regression' or model.name='diffusion'."
        )


if __name__ == "__main__":
    main()