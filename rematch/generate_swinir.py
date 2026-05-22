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
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.experimental.models.diffusion.preconditioning import (
    tEDMPrecondSuperRes,
)
from physicsnemo.models.diffusion.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.models.diffusion.sampling import (
    deterministic_sampler,
    stochastic_sampler,
)
from physicsnemo.models.diffusion.corrdiff_utils import (
    get_time_from_range,
    diffusion_step,
)

from third_party.helpers.generate_helpers import (
    get_dataset_and_sampler,
    get_dataset_and_sampler_timeindices,
    CustomNetCDFWriter,
    custom_save_images,
)
from third_party.datasets.dataset import register_dataset
from rematch.train_swinir import build_swinir_from_cfg
def load_model_checkpoint(checkpoint_path, model, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    step = ckpt.get("step", None) if isinstance(ckpt, dict) else None
    return step


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

    # Parse the inference input times
    # if cfg.generation.times_range and cfg.generation.times:
    #     raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    if cfg.generation.times:
        # print(f"cfg.generation.times given: {cfg.generation.times}")
        times = cfg.generation.times
    else:
        # print(f"no cfg.generation.times given")
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
    
    # class swinir_args:
    #     pass
    # swinir_args.in_chans = 14
    # swinir_args.out_chans = 10
    # swinir_args.img_size = (21,21)
    # swinir_args.upscale = 8
    # swinir_args.window_size = 7
    # swinir_args.embed_dim = 180
    # swinir_args.depths = [6,6,6,6,6,6]
    # swinir_args.num_heads = [6,6,6,6,6,6]
    # swinir_args.mlp_ratio = 2.0
    # swinir_args.drop_path_rate = 0.0
    # swinir_args.multi_step_sampler = True
    # swinir_args.use_checkpoint = False
    model_swinir = build_swinir_from_cfg(cfg).to(device)
    # load_model_checkpoint("/home/nvidia/projects/corrdiff/sr_baselines/swinir_m/checkpoints/swinir_step_00180000.pt", model_swinir, device)
    load_model_checkpoint(cfg.generation.io.reg_ckpt_filename, model_swinir, device)
    print(f"Loaded SWINIR checkpoint from {cfg.generation.io.reg_ckpt_filename}")
    model_swinir.eval()

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()
        if net_res:
            net_res = torch.compile(net_res)
    sampler_fn = partial(stochastic_sampler)
    
    
    # Main generation definition
    def generate_fn():
        with nvtx.annotate("generate_fn", color="green"):
            diffusion_step_kwargs = {}
            img_lr = image_lr.to(memory_format=torch.channels_last)

            img_lr = img_lr.expand(
                cfg.generation.seed_batch_size, -1, -1, -1
            ).to(memory_format=torch.channels_last)
            B, C, H, W = img_lr.shape
            # if C == 16:
            #     # remove two zero channels from the end of the tensor
            #     img_lr_res = img_lr[:, :-2, :, :]
            # else:
            #     # add two zero channels to the end of the tensor
            #     img_lr = torch.cat([img_lr, torch.zeros_like(img_lr[:, :2, :, :])], dim=1)

            # if net_reg:
            #     with nvtx.annotate("regression_model", color="yellow"):
            #         image_reg = regression_step(
            #             net=net_reg,
            #             img_lr=img_lr,
            #             latents_shape=(
            #                 sum(map(len, rank_batches)),
            #                 img_out_channels,
            #                 img_shape[0],
            #                 img_shape[1],
            #             ),  # (batch_size, C, H, W)
            #             lead_time_label=lead_time_label,
            #         )
            # else:
            #     logger0.warning("Regression is None")
            #     image_reg = None
            image_reg = model_swinir(image_lr_raw)
            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    # print(f"img_lr shape: {img_lr.shape}")
                    # print(f"img_shape: {img_shape}")
                    # print(f"img_out_channels: {img_out_channels}")
                    # print(f"mean_hr shape: {mean_hr.shape}")

                    # print(f"img_lr last two channel : {img_lr[:, -4:-2:, :, :]}")
                    # print(f"img_lr shape: {img_lr.shape}")
                    # print(f"img_lr_for_res shape: {img_lr_res.shape}")
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        # img_lr=image_tar-mean_hr,
                        img_lr=img_lr.expand(
                            cfg.generation.seed_batch_size, -1, -1, -1
                        ).to(memory_format=torch.channels_last),
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
                image_reg = None
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
                image_reg=image_reg.repeat(cfg.generation.num_ensembles, 1, 1, 1)
                image_out = image_out.detach() if image_out is not None else None
                image_res = image_res.detach() if image_res is not None else None
                image_reg = image_reg.detach() if image_reg is not None else None

                return image_out, image_res, image_reg


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
            for dataset_index, (image_tar,image_lr_raw, image_lr, *lead_time_label) in zip(
                sampler,
                iter(data_loader),
            ):
                time_index += 1
                if dist.rank == 0:
                    logger0.info(f"starting index: {time_index}")

                if time_index == warmup_steps:
                    start.record()

                image_lr = (
                    image_lr.to(device=device)
                    .to(torch.float32)
                    # .to(memory_format=torch.channels_last)
                )
                image_lr_raw = (
                    image_lr_raw.to(device=device)
                    .to(torch.float32)
                    # .to(memory_format=torch.channels_last)
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
