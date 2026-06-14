# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import torch
import torch._dynamo
from torch.distributed import gather
import numpy as np
import nvtx
import netCDF4 as nc
import math
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo import Module
from physicsnemo.utils.diffusion import stochastic_sampler
from physicsnemo.utils.corrdiff import (
    get_time_from_range,
    regression_step,
    diffusion_step,
)

from third_party.helpers.generate_helpers import (
    get_dataset_and_sampler,
    get_dataset_and_sampler_timeindices,
    CustomNetCDFWriter,
    custom_save_images,
)
from third_party.datasets.dataset import register_dataset

from third_party.baselines.uncertainty_quantification.uq_loss import VerifierQuantileScalarResidualLoss, ResidualLossBiasCorrector_rmse
@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # Handle the batch size
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    if cfg.generation.times:
        times = cfg.generation.times
    else:
        times = None

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    # Register dataset (if custom dataset)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")
    load_all_times = getattr(cfg.generation, "load_all_times", False)
    if cfg.dataset.type == "blastnet":
        dataset, sampler = get_dataset_and_sampler_timeindices(
            dataset_cfg=dataset_cfg, load_all_times=load_all_times, time_indices=times
        )
        save_time=False
        save_data_index=True
        cfg.generation.perf.io_synchronous = False
        if load_all_times:
            times = dataset.time()
    else:
        dataset, sampler = get_dataset_and_sampler(
            dataset_cfg=dataset_cfg, times=times, load_all_times=load_all_times
        )
        if load_all_times:
            times = dataset.time()
        save_time=True
        save_data_index=False
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())


    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        logger0.info("ITS REGRESSION")
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        logger0.info("ITS DIFFUSION")
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        logger0.info("ITS BOTH")
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        logger0.info("ITS REGRESSION")
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        logger0.info("ITS DIFFUSION")
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        logger0.info("ITS BOTH")
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(
            to_absolute_path(res_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_res.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_res.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)

        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(
            to_absolute_path(reg_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_reg.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_reg.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)

        bias_corrector_ckpt_filename = cfg.generation.io.bias_ckpt_filename
        logger0.info(f'Loading network from "{bias_corrector_ckpt_filename}"...')
        net_bias = Module.from_checkpoint(
            to_absolute_path(bias_corrector_ckpt_filename),
            override_args={
                "use_apex_gn": getattr(cfg.generation.perf, "use_apex_gn", False)
            },
        )
        net_bias.profile_mode = getattr(cfg.generation.perf, "profile_mode", False)
        net_bias.use_fp16 = getattr(cfg.generation.perf, "use_fp16", False)
        net_bias = net_bias.eval().to(device).to(memory_format=torch.channels_last)
        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None
    uq_type = getattr(cfg.uq, "type", "rmse")
    if uq_type == "rmse":
        loss_fn = ResidualLossBiasCorrector_rmse(
            regression_net=net_reg,
            bias_net=net_bias,
            hr_mean_conditioning=cfg.generation.hr_mean_conditioning,
        )
    else:
        loss_fn = VerifierQuantileScalarResidualLoss(
            regression_net=net_reg,
            bias_net=net_bias,
            hr_mean_conditioning=cfg.generation.hr_mean_conditioning,
        )
    uq_step = loss_fn.bias_correction_step

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
        if net_reg:
            net_reg = torch.compile(net_reg)
    sampler_fn = partial(stochastic_sampler)
    
    # Main generation definition
    def generate_fn():
        with nvtx.annotate("generate_fn", color="green"):
            diffusion_step_kwargs = {}

            # (1, C, H, W)
            img_lr = image_lr.to(memory_format=torch.channels_last)

            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=(
                            sum(map(len, rank_batches)),
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),  # (batch_size, C, H, W)
                    )
                    y_lr = img_lr.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last)
                    
                    pred_uq1, pred_uq2 = uq_step(
                        img_lr= y_lr,
                        img_reg=image_reg[0:1],
                    )
                    B,C,H,W = image_reg.shape
                    pred_uq1_map = pred_uq1[:, :, None, None].expand(-1, -1, H, W)
                    pred_uq2_map  = pred_uq2[:, :, None, None].expand(-1, -1, H, W)
                    
                    # print(f"y_lr.shape:{y_lr.shape}, pred_uq1_map : {pred_uq1_map.shape}, pred_uq1.shape:{pred_uq1.shape}")
                    if uq_type == "rmse":
                        y_lr = torch.cat((y_lr, pred_uq1_map), dim=1)
                    
                    else:
                        y_lr = torch.cat((y_lr, pred_uq1_map, pred_uq2_map), dim=1)

            else:
                logger0.warning("Regression is None")
                image_reg = None
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=y_lr,
                        rank=dist.rank,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                        **diffusion_step_kwargs,
                    )
                    
            if cfg.generation.inference_mode == "regression":
                image_out = image_reg 
                image_res = None
            elif cfg.generation.inference_mode == "diffusion":
                logger0.info(f"Using inference mode: diffusion")
                image_out = image_res
            else:
                image_out = image_reg + image_res


            # Gather tensors on rank 0
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(
                            image_out, dtype=image_out.dtype, device=image_out.device
                        )
                        for _ in range(dist.world_size)
                    ]

                    gathered_reg_tensors = [
                        torch.zeros_like(
                            image_reg, dtype=image_reg.dtype, device=image_reg.device
                        )
                        for _ in range(dist.world_size)
                    ]

                    gathered_res_tensors = [
                        torch.zeros_like(
                            image_res, dtype=image_res.dtype, device=image_res.device
                        )
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_tensors = None
                    gathered_reg_tensors= None
                    gathered_res_tensors=None

                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if dist.rank == 0 else None,
                    dst=0,
                )
                gather(
                    image_reg,
                    gather_list=gathered_reg_tensors if dist.rank == 0 else None,
                    dst=0,
                )
                gather(
                    image_res,
                    gather_list=gathered_res_tensors if dist.rank == 0 else None,
                    dst=0,
                )

                if dist.rank == 0:
                    return torch.cat(gathered_tensors), torch.cat(gathered_res_tensors), torch.cat(gathered_reg_tensors)
                else:
                    logger0.info('In case where img out is none')
                    return None, image_res, image_reg
            else:
                return image_out, image_res, image_reg
        return

    # generate images
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2)
    # Generates model predictions from the input data using the specified
    # `generate_fn`, and save the predictions to the provided NetCDF file. It iterates
    # through the dataset using a data loader, computes predictions, and saves them along
    # with associated metadata.
    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        # add attributes
        f.cfg = str(cfg)

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch_cuda_profiler:
        with torch_nvtx_profiler:
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                writer = CustomNetCDFWriter(
                    f,
                    lat=dataset.latitude(),
                    lon=dataset.longitude(),
                    input_channels=dataset.input_channels(),
                    output_channels=dataset.output_channels(),
                    save_time=save_time,
                    save_data_index=save_data_index,
                )

                if cfg.generation.perf.io_synchronous:
                    writer_executor = ThreadPoolExecutor(
                        max_workers=cfg.generation.perf.num_writer_workers
                    )
                    writer_threads = []

            # Create timer objects only if CUDA is available
            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            else:
                # Dummy no-op functions for CPU case
                class DummyEvent:
                    def record(self):
                        pass

                    def synchronize(self):
                        pass

                    def elapsed_time(self, _):
                        return 0

                start = end = DummyEvent()

            times = dataset.time()
            for dataset_index, (image_tar, image_lr_raw, image_lr, *lead_time_label) in zip(
                sampler,
                iter(data_loader),
            ):
                time_index += 1
                if dist.rank == 0:
                    logger0.info(f"starting index: {time_index}")

                if time_index == warmup_steps:
                    start.record()

                # continue
                image_lr = (
                    image_lr.to(device=device)
                    .to(torch.float32)
                    .to(memory_format=torch.channels_last)
                )
                image_tar = image_tar.to(device=device).to(torch.float32)
                image_out, image_res, image_reg = generate_fn()
                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    if cfg.generation.perf.io_synchronous:
                        # write out data in a seperate thread so we don't hold up inferencing
                        writer_threads.append(
                            writer_executor.submit(
                                custom_save_images,
                                writer,
                                dataset,
                                list(times),
                                image_out.cpu(),
                                image_tar.cpu(),
                                image_lr.cpu(),
                                time_index,
                                dataset_index,
                                image_res=image_res.cpu() if image_res is not None else None,
                                image_reg=image_reg.cpu() if image_reg is not None else None,
                                save_time=save_time,
                                save_data_index=save_data_index,
                            )
                        )
                    else:
                        custom_save_images(
                            writer,
                            dataset,
                            list(times),
                            image_out.cpu(),
                            image_tar.cpu(),
                            image_lr.cpu(),
                            time_index,
                            dataset_index,
                            image_res=image_res.cpu() if image_res is not None else None,
                            image_reg=image_reg.cpu() if image_reg is not None else None,
                            save_time=save_time,
                            save_data_index=save_data_index,
                        )
            end.record()
            end.synchronize()
            elapsed_time = (
                start.elapsed_time(end) / 1000.0 if use_cuda_timing else 0
            )  # Convert ms to s
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0 and use_cuda_timing:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(
                    f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s"
                )
                logger.info(
                    f"Average time per batch element = {average_time_per_batch_element} s"
                )

            # make sure all the workers are done writing
            if dist.rank == 0 and cfg.generation.perf.io_synchronous:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
