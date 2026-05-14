import os
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Union, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDFDataset

import matplotlib.pyplot as plt
ROOT = Path("/data/blastnet_momentum/dataset")

VARS = ["RHO_kgm-3", "UX_ms-1", "UY_ms-1", "UZ_ms-1"]

CHANNEL_NAMES = ["RHO", "UX", "UY", "UZ"]




# ============================================================
# File indexing / reading
# ============================================================

def parse_var_and_id(path: Path):
    stem = path.stem
    var, sample_id = stem.split("_id", 1)
    return var, sample_id


def index_files(root, split="train", level="HR"):
    d = Path(root) / level / split
    index = defaultdict(dict)

    for f in d.glob("*.dat"):
        var, sample_id = parse_var_and_id(f)
        index[sample_id][var] = f

    complete = {
        sample_id: files
        for sample_id, files in index.items()
        if all(v in files for v in VARS)
    }

    return complete


def read_one_variable(path: Path, resolution: int):
    arr = np.fromfile(path, dtype="<f4")
    expected = resolution ** 3

    if arr.size != expected:
        raise ValueError(
            f"Unexpected size for {path}\n"
            f"got {arr.size}, expected {expected}"
        )

    return arr.reshape(resolution, resolution, resolution)


def read_blastnet_sample(files_by_var: Dict[str, Path], resolution: int):
    arrays = []

    for var in VARS:
        arr = read_one_variable(files_by_var[var], resolution)
        arrays.append(arr)

    return np.stack(arrays, axis=0)  # (C, Z, Y, X)


# ============================================================
# Dataset
# ============================================================

class BLASTNetMomentumDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        lr_level="LR_16x",
        metadata_csv: Optional[str] = None,
        kaggle_data_prefix: Optional[str] = None,
        kaggle_data_id: Optional[str] = None,
        kmeans_cluster_idx: Optional[Union[int, Sequence[int]]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.lr_level = lr_level
        self.meta_by_id = {}

        self.hr_index = index_files(self.root, split=split, level="HR")
        self.lr_index = index_files(self.root, split=split, level=lr_level)

        ids = sorted(set(self.hr_index) & set(self.lr_index))

        if metadata_csv is not None:
            meta = pd.read_csv(metadata_csv)
            meta["hash"] = meta["hash"].astype(str)

            if kaggle_data_prefix is not None:
                meta = meta[
                    meta["kaggle_data_id"].str.startswith(kaggle_data_prefix)
                ]

            if kaggle_data_id is not None:
                meta = meta[
                    meta["kaggle_data_id"] == kaggle_data_id
                ]

            if kmeans_cluster_idx is not None:
                if isinstance(kmeans_cluster_idx, int):
                    cluster_ids = [kmeans_cluster_idx]
                else:
                    cluster_ids = list(kmeans_cluster_idx)

                meta = meta[
                    meta["kmeans_cluster_idx"].isin(cluster_ids)
                ]

            allowed_ids = set(meta["hash"])
            ids = [sid for sid in ids if sid in allowed_ids]

            self.meta_by_id = (
                meta.set_index("hash")[
                    ["kaggle_data_id", "case_description", "kmeans_cluster_idx"]
                ]
                .to_dict(orient="index")
            )

        self.ids = ids

        if len(self.ids) == 0:
            raise RuntimeError("No paired HR/LR samples found after filtering.")

        if lr_level == "LR_8x":
            self.lr_res = 16
        elif lr_level == "LR_16x":
            self.lr_res = 8
        elif lr_level == "LR_32x":
            self.lr_res = 4
        else:
            raise ValueError(f"Unknown lr_level: {lr_level}")

        print(f"Loaded {len(self.ids)} paired samples")
        print(f"split: {split}")
        print(f"lr_level: {lr_level}")
        print(f"lr_res: {self.lr_res}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        hr = read_blastnet_sample(
            self.hr_index[sample_id],
            resolution=128,
        )  # (C, 128, 128, 128)

        lr = read_blastnet_sample(
            self.lr_index[sample_id],
            resolution=self.lr_res,
        )  # (C, Z_lr, Y_lr, X_lr)

        meta = self.meta_by_id.get(sample_id, {})

        return {
            "lr": torch.from_numpy(lr).float(),
            "hr": torch.from_numpy(hr).float(),
            "id": sample_id,
            "kaggle_data_id": meta.get("kaggle_data_id", "unknown"),
            "case_description": meta.get("case_description", "unknown"),
            "kmeans_cluster_idx": meta.get("kmeans_cluster_idx", "unknown"),
        }


# ============================================================
# Normalization / conversion
# ============================================================

def normalize_lr_hr_matched_slices(
    lr: torch.Tensor,
    hr: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Parameters
    ----------
    lr : torch.Tensor
        Shape (C, Z_lr, Y_lr, X_lr), e.g. (4, 8, 8, 8) for LR_16x.
    hr : torch.Tensor
        Shape (C, 128, 128, 128).

    Returns
    -------
    lr_norm : torch.Tensor
        Shape (C, Z_lr, Y_lr, X_lr).
    hr_matched_norm : torch.Tensor
        Shape (C, Z_lr, 128, 128).
        Only matched HR slices are retained.
    """

    C, Z_lr, Y_lr, X_lr = lr.shape
    C_hr, Z_hr, Y_hr, X_hr = hr.shape

    assert C == C_hr
    assert Y_hr == 128 and X_hr == 128
    assert Z_hr % Z_lr == 0

    ratio = Z_hr // Z_lr

    lr_norm = torch.empty_like(lr)
    hr_matched_norm = torch.empty((C, Z_lr, Y_hr, X_hr), dtype=hr.dtype)

    for c in range(C):
        for z_lr in range(Z_lr):
            z_hr = z_lr * ratio

            lr_slice = lr[c, z_lr]          # (Y_lr, X_lr)
            hr_slice = hr[c, z_hr]          # (128, 128)

            combined_min = torch.minimum(lr_slice.min(), hr_slice.min())
            combined_max = torch.maximum(lr_slice.max(), hr_slice.max())
            denom = combined_max - combined_min

            if denom < eps:
                lr_norm[c, z_lr] = 0.0
                hr_matched_norm[c, z_lr] = 0.0
            else:
                lr_norm[c, z_lr] = (lr_slice - combined_min) / denom
                hr_matched_norm[c, z_lr] = (hr_slice - combined_min) / denom

    return lr_norm, hr_matched_norm


def upsample_lr_slices_to_hr(
    lr_norm: torch.Tensor,
    target_size=(128, 128),
):
    """
    Convert normalized LR slices to HR-size 2D input.

    Parameters
    ----------
    lr_norm : torch.Tensor
        Shape (C, Z_lr, Y_lr, X_lr).

    Returns
    -------
    lr_up : torch.Tensor
        Shape (Z_lr, C, 128, 128).
    """

    C, Z_lr, Y_lr, X_lr = lr_norm.shape

    # (C, Z, Y, X) -> (Z, C, Y, X)
    lr_2d = lr_norm.permute(1, 0, 2, 3).contiguous()

    lr_up = F.interpolate(
        lr_2d,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return lr_up  # (Z_lr, C, 128, 128)


def make_output_hr_slices(
    hr_matched_norm: torch.Tensor,
):
    """
    Convert matched HR slices to flattened 2D output.

    Parameters
    ----------
    hr_matched_norm : torch.Tensor
        Shape (C, Z_lr, 128, 128).

    Returns
    -------
    hr_2d : torch.Tensor
        Shape (Z_lr, C, 128, 128).
    """

    return hr_matched_norm.permute(1, 0, 2, 3).contiguous()


# ============================================================
# NetCDF writer
# ============================================================

def create_blastnet_nc(
    root="/data/blastnet_momentum/dataset",
    split="train",
    lr_level="LR_16x",
    output_nc="/data/blastnet_momentum/blastnet_train_lr16x_z8_minmax.nc",
    metadata_csv=None,
    kaggle_data_prefix=None,
    kaggle_data_id=None,
    kmeans_cluster_idx=None,
    dtype="float32",
):
    ds = BLASTNetMomentumDataset(
        root=root,
        split=split,
        lr_level=lr_level,
        metadata_csv=metadata_csv,
        kaggle_data_prefix=kaggle_data_prefix,
        kaggle_data_id=kaggle_data_id,
        kmeans_cluster_idx=kmeans_cluster_idx,
    )
    Z_lr = ds.lr_res  # LR_16x -> 8
    C = len(VARS)
    H = 128
    W = 128
    N = len(ds) * Z_lr

    output_nc = Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_nc}")
    print(f"Final input shape : ({N}, {C}, {H}, {W})")
    print(f"Final output shape: ({N}, {C}, {H}, {W})")

    with NetCDFDataset(output_nc, "w", format="NETCDF4") as nc:
        # Root dimensions
        nc.createDimension("sample", N)
        nc.createDimension("channel", C)
        nc.createDimension("y", H)
        nc.createDimension("x", W)

        # Metadata dimensions
        nc.createDimension("source_sample", len(ds))
        nc.createDimension("z_lr", Z_lr)
        nc.createDimension("str_dim", 128)

        # Groups
        g_input = nc.createGroup("input")
        g_output = nc.createGroup("output")
        g_meta = nc.createGroup("metadata")

        # Variables inside groups
        input_var = g_input.createVariable(
            "data",
            dtype,
            ("sample", "channel", "y", "x"),
            zlib=True,
            complevel=4,
            chunksizes=(1, C, H, W),
        )

        output_var = g_output.createVariable(
            "data",
            dtype,
            ("sample", "channel", "y", "x"),
            zlib=True,
            complevel=4,
            chunksizes=(1, C, H, W),
        )

        # Optional metadata
        source_id_var = g_meta.createVariable(
            "source_id",
            str,
            ("source_sample",),
        )
        flat_source_idx_var = g_meta.createVariable(
            "source_sample_index",
            "i4",
            ("sample",),
        )
        flat_z_lr_var = g_meta.createVariable(
            "z_lr_index",
            "i4",
            ("sample",),
        )
        flat_z_hr_var = g_meta.createVariable(
            "z_hr_index",
            "i4",
            ("sample",),
        )

        nc.description = "BLASTNet momentum paired LR/HR dataset converted to 2D CorrDiff-like format."
        nc.lr_level = lr_level
        nc.normalization = "Per sample, per channel, per matched z-slice min-max using paired LR and HR slices."
        nc.input_group = "Upsampled normalized LR slices."
        nc.output_group = "Normalized matched HR slices."
        nc.channel_names = ",".join(CHANNEL_NAMES)
        nc.original_variables = ",".join(VARS)

        write_idx = 0

        for sample_idx in range(len(ds)):
            sample = ds[sample_idx]
            lr = sample["lr"]  # (C, 8, 8, 8)
            hr = sample["hr"]  # (C, 128, 128, 128)

            C_lr, Z_lr_sample, Y_lr, X_lr = lr.shape
            C_hr, Z_hr, Y_hr, X_hr = hr.shape
            ratio = Z_hr // Z_lr_sample

            if Z_lr_sample != Z_lr:
                raise RuntimeError(f"Unexpected Z_lr: {Z_lr_sample}, expected {Z_lr}")

            lr_norm, hr_matched_norm = normalize_lr_hr_matched_slices(lr, hr)

            lr_up = upsample_lr_slices_to_hr(
                lr_norm,
                target_size=(H, W),
            )  # (Z_lr, C, 128, 128)

            hr_2d = make_output_hr_slices(
                hr_matched_norm,
            )  # (Z_lr, C, 128, 128)

            start = write_idx
            end = write_idx + Z_lr

            input_var[start:end, :, :, :] = lr_up.numpy().astype(np.float32)
            output_var[start:end, :, :, :] = hr_2d.numpy().astype(np.float32)

            source_id_var[sample_idx] = sample["id"]

            for z_lr in range(Z_lr):
                z_hr = z_lr * ratio
                flat_idx = start + z_lr

                flat_source_idx_var[flat_idx] = sample_idx
                flat_z_lr_var[flat_idx] = z_lr
                flat_z_hr_var[flat_idx] = z_hr

            write_idx = end

            if (sample_idx + 1) % 25 == 0 or (sample_idx + 1) == len(ds):
                print(
                    f"[{split}] processed {sample_idx + 1}/{len(ds)} samples "
                    f"-> wrote {write_idx}/{N} 2D slices"
                )

    print("Done.")
    print(f"Saved: {output_nc}")

def create_blastnet_nc_multiple_input(
    root="/data/blastnet_momentum/dataset",
    split="train",
    lr_level="LR_8x",
    output_nc="/data/blastnet_momentum/blastnet_train_lr8x_zmix_multiple_input_minmax.nc",
    metadata_csv=None,
    kaggle_data_prefix=None,
    kaggle_data_id=None,
    kmeans_cluster_idx=None,
    dtype="float32",
    z_lr_target: Optional[int] = None,
    level1_sigma: float = 2.0,
    level3_sigma: float = 1.0,
    level3_center: Union[str, int] = "lower",
    zlib: bool = True,
    complevel: int = 4,
):
    """
    Create BLASTNet 2D paired NetCDF with four LR input variants.

    Stored variables
    ----------------
    input/input_level0 : aligned one-hot LR z-slice
    input/input_level1 : Gaussian z-mix centered at target z
    input/input_level2 : uniform z-average
    input/input_level3 : off-center Gaussian z-mix

    output/data        : selected HR target slice

    Design
    ------
    For each original 3D sample, only one 2D pair is written.

    If LR has shape (C, Z_lr, Y_lr, X_lr) and HR has shape
    (C, Z_hr, 128, 128), then

        z0 = Z_lr // 2 - 1
        z_hr_target = z0 * (Z_hr // Z_lr)

    by default.

    Normalization
    -------------
    For each sample and channel, the full LR z-volume and the selected HR
    target slice are normalized together using one common min/max. This avoids
    destroying the meaning of z-mixing.
    """

    ds = BLASTNetMomentumDataset(
        root=root,
        split=split,
        lr_level=lr_level,
        metadata_csv=metadata_csv,
        kaggle_data_prefix=kaggle_data_prefix,
        kaggle_data_id=kaggle_data_id,
        kmeans_cluster_idx=kmeans_cluster_idx,
    )

    Z_lr = ds.lr_res
    C = len(VARS)
    H = 128
    W = 128
    N = len(ds)

    if z_lr_target is None:
        z_lr_target = Z_lr // 2 - 1

    if not (0 <= z_lr_target < Z_lr):
        raise ValueError(
            f"z_lr_target={z_lr_target} is out of range for Z_lr={Z_lr}"
        )

    # Decide level 3 center.
    if isinstance(level3_center, str):
        if level3_center == "lower":
            z_off = 0
        elif level3_center == "upper":
            z_off = Z_lr - 1
        elif level3_center == "far":
            # Choose the farther boundary from z_lr_target.
            dist_lower = abs(z_lr_target - 0)
            dist_upper = abs((Z_lr - 1) - z_lr_target)
            z_off = 0 if dist_lower >= dist_upper else Z_lr - 1
        else:
            raise ValueError(
                "level3_center must be one of {'lower', 'upper', 'far'} "
                f"or an integer. Got {level3_center}."
            )
    else:
        z_off = int(level3_center)

    if not (0 <= z_off < Z_lr):
        raise ValueError(f"z_off={z_off} is out of range for Z_lr={Z_lr}")

    # Fixed z-mixing weights for all samples.
    weights_level0 = make_one_hot_weights(Z_lr, z_lr_target)
    weights_level1 = make_gaussian_z_weights(
        Z=Z_lr,
        center=z_lr_target,
        sigma=level1_sigma,
    )
    weights_level2 = make_uniform_weights(Z_lr)
    weights_level3 = make_gaussian_z_weights(
        Z=Z_lr,
        center=z_off,
        sigma=level3_sigma,
    )

    level_weights = {
        "input_level0": weights_level0,
        "input_level1": weights_level1,
        "input_level2": weights_level2,
        "input_level3": weights_level3,
    }

    output_nc = Path(output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_nc}")
    print(f"Number of source 3D samples: {len(ds)}")
    print(f"Number of written 2D samples: {N}")
    print(f"LR level: {lr_level}")
    print(f"Z_lr: {Z_lr}")
    print(f"z_lr_target: {z_lr_target}")
    print(f"level1_sigma: {level1_sigma}")
    print(f"level3_center: {z_off}")
    print(f"level3_sigma: {level3_sigma}")
    print(f"Final input shapes : ({N}, {C}, {H}, {W}) for each input_level")
    print(f"Final output shape : ({N}, {C}, {H}, {W})")

    with NetCDFDataset(output_nc, "w", format="NETCDF4") as nc:
        # ====================================================
        # Dimensions
        # ====================================================
        nc.createDimension("sample", N)
        nc.createDimension("channel", C)
        nc.createDimension("y", H)
        nc.createDimension("x", W)

        nc.createDimension("source_sample", len(ds))
        nc.createDimension("z_lr", Z_lr)

        # ====================================================
        # Groups
        # ====================================================
        g_input = nc.createGroup("input")
        g_output = nc.createGroup("output")
        g_meta = nc.createGroup("metadata")
        g_weights = nc.createGroup("z_mixing_weights")

        # ====================================================
        # Input variables
        # ====================================================
        input_vars = {}

        for level_name in [
            "input_level0",
            "input_level1",
            "input_level2",
            "input_level3",
        ]:
            input_vars[level_name] = g_input.createVariable(
                level_name,
                dtype,
                ("sample", "channel", "y", "x"),
                zlib=zlib,
                complevel=complevel,
                chunksizes=(1, C, H, W),
            )

            input_vars[level_name].long_name = level_name
            input_vars[level_name].dimensions_description = (
                "sample, channel, y, x"
            )

        input_vars["input_level0"].description = (
            "Aligned LR slice: one-hot z-mixing at z_lr_target."
        )
        input_vars["input_level1"].description = (
            "Target-centered Gaussian z-mixing."
        )
        input_vars["input_level2"].description = (
            "Uniform z-average over all LR z slices."
        )
        input_vars["input_level3"].description = (
            "Off-center Gaussian z-mixing, localized away from target z."
        )

        # Optional compatibility alias.
        # Some existing loaders may expect input/data.
        # Here we do NOT create input/data to avoid ambiguity.
        g_input.default_variable = "input_level0"
        g_input.available_variables = ",".join(input_vars.keys())

        # ====================================================
        # Output variable
        # ====================================================
        output_var = g_output.createVariable(
            "data",
            dtype,
            ("sample", "channel", "y", "x"),
            zlib=zlib,
            complevel=complevel,
            chunksizes=(1, C, H, W),
        )
        output_var.long_name = "selected HR target slice"
        output_var.description = (
            "Normalized HR slice paired with all input levels."
        )

        # ====================================================
        # Metadata variables
        # ====================================================
        source_id_var = g_meta.createVariable(
            "source_id",
            str,
            ("source_sample",),
        )

        flat_source_idx_var = g_meta.createVariable(
            "source_sample_index",
            "i4",
            ("sample",),
        )

        flat_z_lr_var = g_meta.createVariable(
            "z_lr_index",
            "i4",
            ("sample",),
        )

        flat_z_hr_var = g_meta.createVariable(
            "z_hr_index",
            "i4",
            ("sample",),
        )

        norm_min_var = g_meta.createVariable(
            "norm_min",
            dtype,
            ("sample", "channel"),
            zlib=zlib,
            complevel=complevel,
        )

        norm_max_var = g_meta.createVariable(
            "norm_max",
            dtype,
            ("sample", "channel"),
            zlib=zlib,
            complevel=complevel,
        )

        # Optional metadata from CSV.
        kaggle_data_id_var = g_meta.createVariable(
            "kaggle_data_id",
            str,
            ("source_sample",),
        )
        case_description_var = g_meta.createVariable(
            "case_description",
            str,
            ("source_sample",),
        )
        cluster_idx_var = g_meta.createVariable(
            "kmeans_cluster_idx",
            str,
            ("source_sample",),
        )

        # ====================================================
        # Store z-mixing weights
        # ====================================================
        for level_name, weights in level_weights.items():
            w_var = g_weights.createVariable(
                level_name,
                dtype,
                ("z_lr",),
            )
            w_var[:] = weights.astype(np.float32)
            w_var.description = f"z-mixing weights for {level_name}"
            w_var.sum = float(weights.sum())

        # ====================================================
        # Root attributes
        # ====================================================
        nc.description = (
            "BLASTNet Momentum paired LR/HR dataset converted to 2D "
            "CorrDiff-like format with multiple structured LR z-mixing inputs."
        )
        nc.dataset_type = "BLASTNet Momentum 2D z-mixing multiple-input dataset"
        nc.split = split
        nc.lr_level = lr_level
        nc.channel_names = ",".join(CHANNEL_NAMES)
        nc.original_variables = ",".join(VARS)

        nc.num_source_samples = int(len(ds))
        nc.num_written_samples = int(N)

        nc.hr_resolution = int(128)
        nc.lr_resolution = int(Z_lr)
        nc.z_lr_target = int(z_lr_target)
        nc.level1_sigma = float(level1_sigma)
        nc.level2_type = "uniform"
        nc.level3_center = int(z_off)
        nc.level3_sigma = float(level3_sigma)

        nc.input_levels = (
            "input_level0,input_level1,input_level2,input_level3"
        )

        nc.input_level0_description = (
            "One-hot aligned LR z-slice at z_lr_target."
        )
        nc.input_level1_description = (
            "Gaussian z-mixing centered at z_lr_target."
        )
        nc.input_level2_description = (
            "Uniform average over all LR z-slices."
        )
        nc.input_level3_description = (
            "Narrow off-center Gaussian z-mixing centered at z_off."
        )

        nc.normalization = (
            "Per source sample and channel, use common min/max over the full "
            "LR z-volume and the selected HR target slice. The same normalized "
            "LR volume is used to construct all z-mixed input levels."
        )

        nc.output_group = "output/data"
        nc.metadata_group = "metadata"
        nc.z_mixing_weights_group = "z_mixing_weights"

        if metadata_csv is not None:
            nc.metadata_csv = str(metadata_csv)
        if kaggle_data_prefix is not None:
            nc.kaggle_data_prefix = str(kaggle_data_prefix)
        if kaggle_data_id is not None:
            nc.kaggle_data_id_filter = str(kaggle_data_id)
        if kmeans_cluster_idx is not None:
            nc.kmeans_cluster_idx_filter = str(kmeans_cluster_idx)

        # ====================================================
        # Write samples
        # ====================================================
        write_idx = 0

        for sample_idx in range(len(ds)):
            sample = ds[sample_idx]
            lr = sample["lr"]  # (C, Z_lr, Y_lr, X_lr)
            hr = sample["hr"]  # (C, 128, 128, 128)

            C_lr, Z_lr_sample, Y_lr, X_lr = lr.shape
            C_hr, Z_hr, Y_hr, X_hr = hr.shape

            if C_lr != C:
                raise RuntimeError(f"Unexpected LR C: {C_lr}, expected {C}")
            if C_hr != C:
                raise RuntimeError(f"Unexpected HR C: {C_hr}, expected {C}")
            if Z_lr_sample != Z_lr:
                raise RuntimeError(
                    f"Unexpected Z_lr: {Z_lr_sample}, expected {Z_lr}"
                )
            if Y_hr != H or X_hr != W:
                raise RuntimeError(
                    f"Unexpected HR spatial size: {(Y_hr, X_hr)}, expected {(H, W)}"
                )
            if Z_hr % Z_lr_sample != 0:
                raise RuntimeError(
                    f"Z_hr={Z_hr} is not divisible by Z_lr={Z_lr_sample}"
                )

            ratio = Z_hr // Z_lr_sample
            z_hr_target = z_lr_target * ratio

            if not (0 <= z_hr_target < Z_hr):
                raise RuntimeError(
                    f"z_hr_target={z_hr_target} out of range for Z_hr={Z_hr}"
                )

            # Select fixed HR target slice.
            hr_target = hr[:, z_hr_target, :, :]  # (C, 128, 128)

            # Normalize full LR volume and selected HR target together.
            lr_norm, hr_target_norm, norm_min, norm_max = (
                normalize_lr_volume_and_target_hr_slice(
                    lr=lr,
                    hr_target=hr_target,
                )
            )

            # Write each input level.
            for level_name, weights in level_weights.items():
                lr_mixed = mix_lr_z_slices(
                    lr_norm=lr_norm,
                    weights=weights,
                )  # (C, Y_lr, X_lr)

                lr_up = upsample_single_lr_slice_to_hr(
                    lr_mixed,
                    target_size=(H, W),
                )  # (C, 128, 128)

                input_vars[level_name][write_idx, :, :, :] = (
                    lr_up.detach().cpu().numpy().astype(np.float32)
                )

            # Write output.
            output_var[write_idx, :, :, :] = (
                hr_target_norm.detach().cpu().numpy().astype(np.float32)
            )

            # Write metadata.
            source_id_var[sample_idx] = sample["id"]
            kaggle_data_id_var[sample_idx] = str(
                sample.get("kaggle_data_id", "unknown")
            )
            case_description_var[sample_idx] = str(
                sample.get("case_description", "unknown")
            )
            cluster_idx_var[sample_idx] = str(
                sample.get("kmeans_cluster_idx", "unknown")
            )

            flat_source_idx_var[write_idx] = sample_idx
            flat_z_lr_var[write_idx] = int(z_lr_target)
            flat_z_hr_var[write_idx] = int(z_hr_target)

            norm_min_var[write_idx, :] = (
                norm_min.detach().cpu().numpy().astype(np.float32)
            )
            norm_max_var[write_idx, :] = (
                norm_max.detach().cpu().numpy().astype(np.float32)
            )

            write_idx += 1

            if (sample_idx + 1) % 25 == 0 or (sample_idx + 1) == len(ds):
                print(
                    f"[{split}] processed {sample_idx + 1}/{len(ds)} samples "
                    f"-> wrote {write_idx}/{N} 2D samples"
                )

        if write_idx != N:
            raise RuntimeError(f"write_idx={write_idx}, expected N={N}")

    print("Done.")
    print(f"Saved: {output_nc}")

from pathlib import Path
import numpy as np
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset


def split_blastnet_multiple_input_nc(
    input_nc,
    train_nc,
    calibration_nc,
    train_fraction=2 / 3,
    dtype="float32",
    zlib=True,
    complevel=4,
):
    """
    Split a multiple-input BLASTNet NetCDF file into train/calibration NetCDF files.

    This preserves:
        /input/input_level0
        /input/input_level1
        /input/input_level2
        /input/input_level3
        /output/data
        /metadata variables that depend on sample
        /z_mixing_weights variables

    Parameters
    ----------
    input_nc : str or Path
        Source NetCDF file.
    train_nc : str or Path
        Output NetCDF file for first train_fraction samples.
    calibration_nc : str or Path
        Output NetCDF file for remaining samples.
    train_fraction : float
        Fraction of samples assigned to train split.
    """

    input_nc = Path(input_nc)
    train_nc = Path(train_nc)
    calibration_nc = Path(calibration_nc)

    train_nc.parent.mkdir(parents=True, exist_ok=True)
    calibration_nc.parent.mkdir(parents=True, exist_ok=True)

    ds_input = xr.open_dataset(input_nc, group="input")
    ds_output = xr.open_dataset(input_nc, group="output")
    ds_meta = xr.open_dataset(input_nc, group="metadata")
    ds_weights = xr.open_dataset(input_nc, group="z_mixing_weights")

    n_samples = ds_input.sizes["sample"]
    n_channels = ds_input.sizes["channel"]
    H = ds_input.sizes["y"]
    W = ds_input.sizes["x"]

    split_idx = int(n_samples * train_fraction)

    splits = {
        "train": {
            "path": train_nc,
            "sample_slice": slice(0, split_idx),
        },
        "calibration": {
            "path": calibration_nc,
            "sample_slice": slice(split_idx, n_samples),
        },
    }

    input_level_names = [
        name for name in ds_input.data_vars
        if name.startswith("input_level")
    ]
    input_level_names = sorted(input_level_names)

    if len(input_level_names) == 0:
        raise RuntimeError("No input_level* variables found in /input group.")

    for split_name, split_info in splits.items():
        out_path = split_info["path"]
        sample_slice = split_info["sample_slice"]

        sample_indices = np.arange(n_samples)[sample_slice]
        n_out = len(sample_indices)

        print(f"Writing {split_name}: {out_path}")
        print(f"  samples: {sample_indices[0] if n_out > 0 else 'none'} "
              f"to {sample_indices[-1] if n_out > 0 else 'none'}")
        print(f"  n_out: {n_out}")

        with NetCDFDataset(out_path, "w", format="NETCDF4") as nc:
            # ====================================================
            # Dimensions
            # ====================================================
            nc.createDimension("sample", n_out)
            nc.createDimension("channel", n_channels)
            nc.createDimension("y", H)
            nc.createDimension("x", W)

            if "source_sample" in ds_meta.sizes:
                nc.createDimension("source_sample", n_out)

            if "z_lr" in ds_weights.sizes:
                nc.createDimension("z_lr", ds_weights.sizes["z_lr"])

            # ====================================================
            # Groups
            # ====================================================
            g_input = nc.createGroup("input")
            g_output = nc.createGroup("output")
            g_meta = nc.createGroup("metadata")
            g_weights = nc.createGroup("z_mixing_weights")

            # ====================================================
            # Copy root attrs
            # ====================================================
            src_root = NetCDFDataset(input_nc, "r")
            for key, value in src_root.__dict__.items():
                setattr(nc, key, value)
            src_root.close()

            nc.parent_file = str(input_nc)
            nc.split_from_parent = split_name
            nc.parent_num_samples = int(n_samples)
            nc.train_fraction = float(train_fraction)
            nc.parent_split_index = int(split_idx)
            nc.num_written_samples = int(n_out)

            # ====================================================
            # Input variables
            # ====================================================
            for level_name in input_level_names:
                var = g_input.createVariable(
                    level_name,
                    dtype,
                    ("sample", "channel", "y", "x"),
                    zlib=zlib,
                    complevel=complevel,
                    chunksizes=(1, n_channels, H, W),
                )

                # copy variable attrs
                for key, value in ds_input[level_name].attrs.items():
                    setattr(var, key, value)

                data = ds_input[level_name].isel(sample=sample_slice).values
                var[:, :, :, :] = data.astype(np.float32)

            g_input.default_variable = ds_input.attrs.get(
                "default_variable",
                "input_level0",
            )
            g_input.available_variables = ",".join(input_level_names)

            # ====================================================
            # Output variable
            # ====================================================
            out_var = g_output.createVariable(
                "data",
                dtype,
                ("sample", "channel", "y", "x"),
                zlib=zlib,
                complevel=complevel,
                chunksizes=(1, n_channels, H, W),
            )

            for key, value in ds_output["data"].attrs.items():
                setattr(out_var, key, value)

            out_data = ds_output["data"].isel(sample=sample_slice).values
            out_var[:, :, :, :] = out_data.astype(np.float32)

            # ====================================================
            # Metadata variables
            # ====================================================
            for meta_name in ds_meta.data_vars:
                meta_da = ds_meta[meta_name]
                dims = meta_da.dims

                # Variables indexed by sample.
                if "sample" in dims:
                    new_dims = tuple(dims)

                    fill_kwargs = {}
                    if meta_da.dtype.kind in ["U", "S", "O"]:
                        var = g_meta.createVariable(meta_name, str, new_dims)
                        values = meta_da.isel(sample=sample_slice).values
                        var[:] = values.astype(str)
                    else:
                        var = g_meta.createVariable(
                            meta_name,
                            meta_da.dtype,
                            new_dims,
                            zlib=zlib,
                            complevel=complevel,
                        )
                        values = meta_da.isel(sample=sample_slice).values
                        var[:] = values

                    for key, value in meta_da.attrs.items():
                        setattr(var, key, value)

                # Variables indexed by source_sample.
                elif "source_sample" in dims:
                    new_dims = tuple(dims)

                    if meta_da.dtype.kind in ["U", "S", "O"]:
                        var = g_meta.createVariable(meta_name, str, new_dims)
                        values = meta_da.isel(source_sample=sample_slice).values
                        var[:] = values.astype(str)
                    else:
                        var = g_meta.createVariable(
                            meta_name,
                            meta_da.dtype,
                            new_dims,
                            zlib=zlib,
                            complevel=complevel,
                        )
                        values = meta_da.isel(source_sample=sample_slice).values
                        var[:] = values

                    for key, value in meta_da.attrs.items():
                        setattr(var, key, value)

                # Other metadata variables: copy as-is if any.
                else:
                    new_dims = tuple(dims)

                    for dim in new_dims:
                        if dim not in nc.dimensions:
                            nc.createDimension(dim, ds_meta.sizes[dim])

                    if meta_da.dtype.kind in ["U", "S", "O"]:
                        var = g_meta.createVariable(meta_name, str, new_dims)
                        var[:] = meta_da.values.astype(str)
                    else:
                        var = g_meta.createVariable(
                            meta_name,
                            meta_da.dtype,
                            new_dims,
                            zlib=zlib,
                            complevel=complevel,
                        )
                        var[:] = meta_da.values

                    for key, value in meta_da.attrs.items():
                        setattr(var, key, value)

            # Re-index metadata inside each split.
            if "source_sample_index" in g_meta.variables:
                g_meta.variables["source_sample_index"][:] = np.arange(
                    n_out,
                    dtype=np.int32,
                )

            # ====================================================
            # z_mixing_weights variables
            # ====================================================
            for weight_name in ds_weights.data_vars:
                w_da = ds_weights[weight_name]
                dims = w_da.dims

                for dim in dims:
                    if dim not in nc.dimensions:
                        nc.createDimension(dim, ds_weights.sizes[dim])

                w_var = g_weights.createVariable(
                    weight_name,
                    dtype,
                    dims,
                )

                w_var[:] = w_da.values.astype(np.float32)

                for key, value in w_da.attrs.items():
                    setattr(w_var, key, value)

    ds_input.close()
    ds_output.close()
    ds_meta.close()
    ds_weights.close()

    print("Done.")
    print(f"Train saved      : {train_nc}")
    print(f"Calibration saved: {calibration_nc}")
# ============================================================
# Multiple-input z-mixing utilities
# ============================================================

def make_one_hot_weights(Z: int, center: int) -> np.ndarray:
    weights = np.zeros(Z, dtype=np.float32)
    weights[center] = 1.0
    return weights


def make_uniform_weights(Z: int) -> np.ndarray:
    weights = np.ones(Z, dtype=np.float32)
    weights /= weights.sum()
    return weights


def make_gaussian_z_weights(
    Z: int,
    center: int,
    sigma: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Gaussian weights along z-axis.

    Parameters
    ----------
    Z : int
        Number of LR z slices.
    center : int
        Gaussian center index.
    sigma : float
        Gaussian standard deviation in LR z-index units.

    Returns
    -------
    weights : np.ndarray
        Shape (Z,), sum to 1.
    """
    if sigma <= 0:
        return make_one_hot_weights(Z, center)

    z = np.arange(Z, dtype=np.float32)
    weights = np.exp(-0.5 * ((z - float(center)) / float(sigma)) ** 2)
    weights = weights.astype(np.float32)

    s = weights.sum()
    if s < eps:
        raise ValueError(
            f"Gaussian weights collapsed. Z={Z}, center={center}, sigma={sigma}"
        )

    weights /= s
    return weights


def normalize_lr_volume_and_target_hr_slice(
    lr: torch.Tensor,
    hr_target: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Normalize LR full z-volume and one HR target slice using a common
    per-sample, per-channel min/max.

    This is important for z-mixing. If each LR z-slice is normalized separately,
    z-averaging no longer represents a physically meaningful z-aggregation.

    Parameters
    ----------
    lr : torch.Tensor
        Shape (C, Z_lr, Y_lr, X_lr).
    hr_target : torch.Tensor
        Shape (C, Y_hr, X_hr).

    Returns
    -------
    lr_norm : torch.Tensor
        Shape (C, Z_lr, Y_lr, X_lr).
    hr_target_norm : torch.Tensor
        Shape (C, Y_hr, X_hr).
    norm_min : torch.Tensor
        Shape (C,).
    norm_max : torch.Tensor
        Shape (C,).
    """
    C, Z_lr, Y_lr, X_lr = lr.shape
    C_hr, Y_hr, X_hr = hr_target.shape

    if C != C_hr:
        raise ValueError(f"Channel mismatch: lr C={C}, hr_target C={C_hr}")

    lr_norm = torch.empty_like(lr)
    hr_target_norm = torch.empty_like(hr_target)

    norm_min = torch.empty(C, dtype=lr.dtype)
    norm_max = torch.empty(C, dtype=lr.dtype)

    for c in range(C):
        lr_c = lr[c]              # (Z_lr, Y_lr, X_lr)
        hr_c = hr_target[c]       # (Y_hr, X_hr)

        combined_min = torch.minimum(lr_c.min(), hr_c.min())
        combined_max = torch.maximum(lr_c.max(), hr_c.max())
        denom = combined_max - combined_min

        norm_min[c] = combined_min
        norm_max[c] = combined_max

        if denom < eps:
            lr_norm[c] = 0.0
            hr_target_norm[c] = 0.0
        else:
            lr_norm[c] = (lr_c - combined_min) / denom
            hr_target_norm[c] = (hr_c - combined_min) / denom

    return lr_norm, hr_target_norm, norm_min, norm_max


def mix_lr_z_slices(
    lr_norm: torch.Tensor,
    weights: np.ndarray,
) -> torch.Tensor:
    """
    Weighted sum of normalized LR z-slices.

    Parameters
    ----------
    lr_norm : torch.Tensor
        Shape (C, Z_lr, Y_lr, X_lr).
    weights : np.ndarray
        Shape (Z_lr,).

    Returns
    -------
    mixed_lr : torch.Tensor
        Shape (C, Y_lr, X_lr).
    """
    C, Z_lr, Y_lr, X_lr = lr_norm.shape

    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (Z_lr,):
        raise ValueError(f"weights shape {weights.shape} does not match Z_lr={Z_lr}")

    weights_t = torch.from_numpy(weights).to(
        device=lr_norm.device,
        dtype=lr_norm.dtype,
    )

    mixed_lr = torch.sum(
        lr_norm * weights_t[None, :, None, None],
        dim=1,
    )

    return mixed_lr  # (C, Y_lr, X_lr)


def upsample_single_lr_slice_to_hr(
    lr_2d: torch.Tensor,
    target_size=(128, 128),
):
    """
    Upsample one mixed LR 2D slice to HR size.

    Parameters
    ----------
    lr_2d : torch.Tensor
        Shape (C, Y_lr, X_lr).

    Returns
    -------
    lr_up : torch.Tensor
        Shape (C, 128, 128).
    """
    lr_up = F.interpolate(
        lr_2d[None, :, :, :],
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    return lr_up

def plot_lr_hr_matched_slices(sample, channel=0, save_path=None, plot_z=2):
    """
    Compare LR_8x z-slices with corresponding HR z-slices.
    LR z index: 0..15
    HR z index: z_lr * 8
    """
    lr = sample["lr"].numpy()  # (4, 16, 16, 16)
    hr = sample["hr"].numpy()  # (4, 128, 128, 128)
    C, Z_lr, Y_lr, X_lr = lr.shape
    C, Z_hr, Y_hr, X_hr = hr.shape
    ratio = int(Z_hr / Z_lr)

    lr_ch = lr[channel]
    hr_ch = hr[channel]
    Z_lr = int(Z_lr/plot_z)
    # fig, axes = plt.subplots(Z_lr, 3, figsize=(6, Z_lr*2), constrained_layout=True)
    fig, axes = plt.subplots(3, Z_lr, figsize=(Z_lr*2,6), constrained_layout=True)

    vmin = min(lr_ch.min(), hr_ch.min())
    vmax = max(lr_ch.max(), hr_ch.max())

    for z_lr in range(Z_lr):
        
        z_hr = z_lr * ratio * plot_z
        lr_slice = torch.from_numpy(lr_ch[z_lr*plot_z]).float()  # (16, 16)

        lr_interpolated = F.interpolate(
            lr_slice[None, None, :, :],   # (1, 1, 16, 16)
            size=(Y_hr, X_hr),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()                  # (128, 128)

        axes[0, z_lr].imshow(lr_ch[z_lr*plot_z], origin="lower", vmin=vmin, vmax=vmax)
        axes[0, z_lr].set_title(f"LR z={z_lr*plot_z}", fontsize=15)
        axes[0, z_lr].axis("off")

        axes[1, z_lr].imshow(lr_interpolated, origin="lower", vmin=vmin, vmax=vmax)
        axes[1, z_lr].set_title(rf"$\widetilde{{\mathrm{{LR}}}},\ z={z_lr * plot_z}$", fontsize=15)
        axes[1, z_lr].axis("off")

        im = axes[2, z_lr].imshow(hr_ch[z_hr], origin="lower", vmin=vmin, vmax=vmax)
        axes[2, z_lr].set_title(f"HR z={z_hr}", fontsize=15)
        axes[2, z_lr].axis("off")

        fig.suptitle(
            f"LR-HR matched slices | channel={CHANNEL_NAMES[channel]}\n"
            f"id={sample['id']} | {sample.get('kaggle_data_id', 'unknown')} | "
            f"{sample.get('case_description', 'unknown')} | "
            f"cluster={sample.get('kmeans_cluster_idx', 'unknown')}",
            fontsize=10,
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.6)
    cbar.ax.tick_params(labelsize=12)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
def parse_var_and_id(path):
    stem = path.stem
    var, sample_id = stem.split("_id", 1)
    return var, sample_id

def index_files(root, split="train", level="HR"):
    d = root / level / split
    index = defaultdict(dict)

    for f in d.glob("*.dat"):
        var, sample_id = parse_var_and_id(f)
        index[sample_id][var] = f

    complete = {
        sample_id: files
        for sample_id, files in index.items()
        if all(v in files for v in VARS)
    }

    return complete


def read_one_variable(path, resolution):
    arr = np.fromfile(path, dtype="<f4")
    expected = resolution ** 3

    if arr.size != expected:
        raise ValueError(
            f"Unexpected size for {path}\n"
            f"got {arr.size}, expected {expected}"
        )

    return arr.reshape(resolution, resolution, resolution)

def read_blastnet_sample(files_by_var, resolution):
    arrays = []

    for var in VARS:
        arr = read_one_variable(files_by_var[var], resolution)
        arrays.append(arr)

    x = np.stack(arrays, axis=0)  # (4, D, D, D)
    return x

def get_cluster_list():
    summary_csv = "/data/blastnet_momentum/train_data_summary.csv"

    df = pd.read_csv(summary_csv)

    clusters = df["kmeans_cluster_idx"]
    return clusters

def load_raw_dataset(kaggle_data_id, cluster_idx, channel_idx, split, plot_interval, lr_level=8):
    # kaggle_data_id = "free-propagating-h2-vit-air-li-case-22"
    # kaggle_data_id = "free-propagating-h2-vit-air-li-case-9"
        
    ds = BLASTNetMomentumDataset(
        root="/data/blastnet_momentum/dataset",
        split=split,
        lr_level=f"LR_{lr_level}x",
        metadata_csv=f"/data/blastnet_momentum/{split}_data_summary.csv",
        kaggle_data_id=kaggle_data_id,
        kmeans_cluster_idx=cluster_idx,
    )
    if ds is None:
        print(f"No paired HR/LR samples found for kaggle_data_id: {kaggle_data_id}, cluster_idx: {cluster_idx}, split: {split}")
        return
    
    print(f"len(ds): {len(ds)}")
    for i in range(0, len(ds), plot_interval):
        sample = ds[i]
        print(
            sample["id"],
            sample["kaggle_data_id"],
            sample["case_description"],
            "cluster=", sample["kmeans_cluster_idx"],
            "RHO min/max=",
            sample["hr"][1].min().item(),
            sample["hr"][1].max().item(),
        )
        if kaggle_data_id is None:
            data_id = "all_clusters"
        else:
            data_id = kaggle_data_id

        plot_lr_hr_matched_slices(
            sample,
            channel=channel_idx,
            save_path=(
                f"../plots/blastnet/{data_id}/cluster_{cluster_idx}/{split}_{CHANNEL_NAMES[channel_idx]}/"
                f"lr_hr_matched_{lr_level}x_{sample['id']}.png"
            ),
            plot_z=2
        )
def check_raw_dataset():
    # kaggle_data_id = "free-propagating-h2-vit-air-li-case-22"
    # kaggle_data_id = "free-propagating-h2-vit-air-li-case-9"
    
    kaggle_data_id=None
    channel_idx = 1
    step=1
    # split_list = ["train", "val", "test"]
    split_list=["test"]
    # cluster_list = get_cluster_list()
    # clusters = sorted(cluster_list.unique().tolist())
    clusters = [11]
    print(f"cluster_list: {clusters}")
    for cluster in clusters:
        print(f"cluster: {cluster}")
        for split in split_list:      
            load_raw_dataset(kaggle_data_id=kaggle_data_id, cluster_idx=cluster, channel_idx=channel_idx, split=split, plot_interval=step)   

# ============================================================
# Main
# ============================================================


def main():
    split = "val"
    cluster_idx = [4,11,13,17]
    cluster_name = "4_11_13_17"

    ''' create dataset for all LR Z levels '''
    # create_blastnet_nc(
    #     root="/data/blastnet_momentum/dataset",
    #     split=split,
    #     # lr_level="LR_16x",
    #     lr_level="LR_8x",
    #     metadata_csv=f"/data/blastnet_momentum/{split}_data_summary.csv",
    #     # output_nc=f"/data/blastnet_momentum/blastnet_cluster_{cluster_name}_{split}_lr16x_z8_paired_minmax.nc",
    #     output_nc=f"/data/blastnet_momentum/blastnet_cluster_{cluster_name}_{split}_lr8x_z4_paired_minmax.nc",
    #     kaggle_data_prefix=None,
    #     kaggle_data_id=None,
    #     kmeans_cluster_idx=cluster_idx,
    # )

    ''' create dataset for single LR Z level with multiple input levels '''
    # create_blastnet_nc_multiple_input(
    #     root="/data/blastnet_momentum/dataset",
    #     split=split,
    #     lr_level="LR_8x",  # LR shape = (16, 16, 16)
    #     metadata_csv=f"/data/blastnet_momentum/{split}_data_summary.csv",
    #     output_nc=(
    #         f"/data/blastnet_momentum/"
    #         f"blastnet_multiple_input_cluster_{cluster_name}_{split}_lr8x.nc"
    #     ),
    #     kaggle_data_prefix=None,
    #     kaggle_data_id=None,
    #     kmeans_cluster_idx=cluster_idx,
    #     z_lr_target=None,      # default: Z_lr // 2 - 1
    #     level1_sigma=2.0,
    #     level3_sigma=1.0,
    #     level3_center="lower", # or "upper", "far", or integer
    # )

    ''' split dataset into train/calibration set for multiple input levels '''
    nc_path = "/data/blastnet_momentum/blastnet_multiple_input_cluster_4_11_13_17_train_lr8x.nc"
    split_blastnet_multiple_input_nc(
        input_nc=nc_path,
        train_nc="/data/blastnet_momentum/split_blastnet_multiple_input_cluster_4_11_13_17_train_lr8x.nc",
        calibration_nc="/data/blastnet_momentum/split_blastnet_multiple_input_cluster_4_11_13_17_calibration_lr8x.nc",
        train_fraction=2 / 3,
    )


if __name__ == "__main__":
    # main()
    check_raw_dataset()

# check level input and output 
# if __name__ == "__main__":
#     import os
#     import xarray as xr
#     import matplotlib.pyplot as plt
#     import numpy as np

#     nc_path = "/data/blastnet_momentum/blastnet_multiple_input_cluster_4_11_13_17_test_lr8x.nc"
#     save_dir = "../plots/blastnet/multiple_input"
#     os.makedirs(save_dir, exist_ok=True)

#     ds_input = xr.open_dataset(nc_path, group="input")
#     ds_output = xr.open_dataset(nc_path, group="output")

#     input_levels = [
#         "input_level0",
#         "input_level1",
#         "input_level2",
#         "input_level3",
#     ]

#     channel_names = ["RHO", "UX", "UY", "UZ"]

#     n_samples = ds_input.sizes["sample"]
#     n_channels = ds_input.sizes["channel"]

#     for i in range(n_samples):
#         fig, axes = plt.subplots(
#             n_channels,
#             5,
#             figsize=(5 * 4, 5 * 4),
#             constrained_layout=True,
#         )

#         for c in range(n_channels):
#             row_imgs = []

#             for level in input_levels:
#                 row_imgs.append(
#                     ds_input[level].isel(sample=i, channel=c).values
#                 )

#             row_imgs.append(
#                 ds_output["data"].isel(sample=i, channel=c).values
#             )

#             vmin = min(img.min() for img in row_imgs)
#             vmax = max(img.max() for img in row_imgs)

#             for k, level in enumerate(input_levels):
#                 ax = axes[c, k]
#                 ax.imshow(row_imgs[k], origin="lower", vmin=vmin, vmax=vmax)
#                 ax.set_title(f"{channel_names[c]} | {level}", fontsize=30)
#                 ax.axis("off")

#             ax = axes[c, 4]
#             ax.imshow(row_imgs[4], origin="lower", vmin=vmin, vmax=vmax)
#             ax.set_title(f"{channel_names[c]} | output", fontsize=30)
#             ax.axis("off")

#         fig.suptitle(f"Sample {i}", fontsize=14)

#         save_path = os.path.join(save_dir, f"sample_{i:04d}.png")
#         fig.savefig(save_path, dpi=200, bbox_inches="tight")
#         plt.close(fig)

#         if (i + 1) % 25 == 0 or (i + 1) == n_samples:
#             print(f"Saved {i + 1}/{n_samples}: {save_path}")