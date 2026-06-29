import os
import json
from pathlib import Path

import numpy as np
import xarray as xr
import torch
import lpips


# -----------------------------
# LPIPS model
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn_alex = lpips.LPIPS(net="alex")
loss_fn_alex.eval()
loss_fn_alex.to(DEVICE)


# -----------------------------
# IO
# -----------------------------

def open_samples(f):
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")
    inp = xr.open_dataset(f, group="input")

    pred = pred.merge(root)
    truth = truth.merge(root)
    inp = inp.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    inp = inp.set_coords(["lon", "lat"])

    return truth, pred, root, inp


# -----------------------------
# Shape handling
# -----------------------------

def dataset_to_ensemble_da(
    pred: xr.Dataset,
    truth: xr.Dataset,
    target_ensemble: int = 12,
):
    """
    Return:
        pred_da:  (time, ensemble, channel, y, x)
        truth_da: (time, ensemble, channel, y, x)

    If pred has no ensemble dim or ensemble=1, repeat/broadcast to target_ensemble.
    If pred has target_ensemble, keep it.
    If pred has more than target_ensemble, take first target_ensemble.
    """

    pred_da = pred.to_array(dim="channel")
    truth_da = truth.to_array(dim="channel")

    # Basic truth shape: (time, channel, y, x)
    truth_da = truth_da.transpose("time", "channel", "y", "x")

    # Prediction may or may not have ensemble dimension.
    if "ensemble" in pred_da.dims:
        pred_da = pred_da.transpose("time", "ensemble", "channel", "y", "x")

        E = pred_da.sizes["ensemble"]

        if E == target_ensemble:
            pass

        elif E == 1:
            # Drop ensemble then re-expand to target_ensemble.
            # This broadcasts the same deterministic prediction 12 times.
            pred_da = pred_da.isel(ensemble=0, drop=True)
            pred_da = pred_da.expand_dims(ensemble=np.arange(target_ensemble))
            pred_da = pred_da.transpose("time", "ensemble", "channel", "y", "x")

        elif E > target_ensemble:
            pred_da = pred_da.isel(ensemble=slice(0, target_ensemble))
            pred_da = pred_da.assign_coords(ensemble=np.arange(target_ensemble))

        else:
            raise ValueError(
                f"Prediction ensemble size is {E}, smaller than target_ensemble={target_ensemble}. "
                f"I do not recommend padding non-deterministic ensembles. "
                f"Use target_ensemble={E} or regenerate samples."
            )

    else:
        # Deterministic prediction with no ensemble dim.
        pred_da = pred_da.transpose("time", "channel", "y", "x")
        pred_da = pred_da.expand_dims(ensemble=np.arange(target_ensemble))
        pred_da = pred_da.transpose("time", "ensemble", "channel", "y", "x")

    # Truth has no ensemble. Broadcast truth to same ensemble coords as pred_da.
    truth_da = truth_da.expand_dims({"ensemble": pred_da.ensemble})
    truth_da = truth_da.transpose("time", "ensemble", "channel", "y", "x")

    # Align coordinates just in case.
    pred_da, truth_da = xr.align(pred_da, truth_da, join="exact")

    assert pred_da.dims == ("time", "ensemble", "channel", "y", "x")
    assert truth_da.dims == ("time", "ensemble", "channel", "y", "x")
    assert pred_da.shape == truth_da.shape, f"{pred_da.shape} vs {truth_da.shape}"

    return pred_da, truth_da


# -----------------------------
# Fixed normalization
# -----------------------------

def normalize_with_stats(
    pred_da: xr.DataArray,
    truth_da: xr.DataArray,
    stats_dict: dict,
    clip_sigma: float = 3.0,
):
    """
    Fixed normalization for fair model comparison.

    Input shape:
        pred_da, truth_da: (T, E, C, H, W)

    Uses stats_dict["output"][channel]["mean/std"].
    Then maps roughly [-clip_sigma, clip_sigma] to [-1, 1].
    """

    channels = [str(ch) for ch in pred_da.channel.values]

    means = np.array(
        [stats_dict["output"][ch]["mean"] for ch in channels],
        dtype=np.float32,
    )

    stds = np.array(
        [stats_dict["output"][ch]["std"] for ch in channels],
        dtype=np.float32,
    )

    means = means.reshape(1, 1, len(channels), 1, 1)
    stds = stds.reshape(1, 1, len(channels), 1, 1)

    pred_arr = pred_da.values.astype(np.float32)
    truth_arr = truth_da.values.astype(np.float32)

    pred_arr = (pred_arr - means) / (stds + 1e-6)
    truth_arr = (truth_arr - means) / (stds + 1e-6)

    # LPIPS expects roughly [-1, 1].
    pred_arr = np.clip(pred_arr / clip_sigma, -1.0, 1.0)
    truth_arr = np.clip(truth_arr / clip_sigma, -1.0, 1.0)

    return pred_arr, truth_arr


# -----------------------------
# LPIPS computation
# -----------------------------

def compute_lpips_e12_fixednorm(
    pred_arr: np.ndarray,
    truth_arr: np.ndarray,
    channel_coords,
    time_coords,
    ensemble_coords,
    batch_size: int = 256,
):
    """
    pred_arr/truth_arr shape:
        (T, E, C, H, W)

    Returns:
        lpips_channel:   (C,)
        lpips_ensemble:  (E,)
        lpips_scalar:    scalar
        lpips_ds:        xarray Dataset containing full d(time, ensemble, channel)
    """

    pred_t = torch.from_numpy(pred_arr).float()
    truth_t = torch.from_numpy(truth_arr).float()

    assert pred_t.shape == truth_t.shape, f"{pred_t.shape} vs {truth_t.shape}"
    assert pred_t.ndim == 5, f"Expected (T,E,C,H,W), got {pred_t.shape}"

    T, E, C, H, W = pred_t.shape

    print(f"LPIPS input shape: T={T}, E={E}, C={C}, H={H}, W={W}")
    print(f"pred range:  {pred_t.min().item():.4f}, {pred_t.max().item():.4f}")
    print(f"truth range: {truth_t.min().item():.4f}, {truth_t.max().item():.4f}")

    # Flatten order is exactly T, E, C.
    # Therefore d.view(T, E, C) is valid.
    pred_flat = pred_t.reshape(T * E * C, H, W).unsqueeze(1)
    truth_flat = truth_t.reshape(T * E * C, H, W).unsqueeze(1)

    assert torch.allclose(pred_t[0, 0, 0], pred_flat[0, 0])
    assert torch.allclose(truth_t[0, 0, 0], truth_flat[0, 0])

    d_chunks = []

    with torch.no_grad():
        for start in range(0, pred_flat.shape[0], batch_size):
            end = min(start + batch_size, pred_flat.shape[0])

            pred_rgb = pred_flat[start:end].repeat(1, 3, 1, 1).to(DEVICE)
            truth_rgb = truth_flat[start:end].repeat(1, 3, 1, 1).to(DEVICE)

            d_batch = loss_fn_alex(pred_rgb, truth_rgb)
            d_batch = d_batch.reshape(-1).detach().cpu()

            d_chunks.append(d_batch)

    d = torch.cat(d_chunks, dim=0)
    d = d.view(T, E, C)

    lpips_channel = d.mean(dim=(0, 1))       # (C,)
    lpips_ensemble = d.mean(dim=(0, 2))      # (E,)
    lpips_time_channel = d.mean(dim=1)       # (T, C)
    lpips_time_ensemble = d.mean(dim=2)      # (T, E)
    lpips_scalar = d.mean().item()

    lpips_full_da = xr.DataArray(
        d.numpy(),
        dims=("time", "ensemble", "channel"),
        coords={
            "time": time_coords,
            "ensemble": ensemble_coords,
            "channel": channel_coords,
        },
        name="lpips_full",
    )

    lpips_time_channel_da = xr.DataArray(
        lpips_time_channel.numpy(),
        dims=("time", "channel"),
        coords={
            "time": time_coords,
            "channel": channel_coords,
        },
        name="lpips_time_channel",
    )

    lpips_time_ensemble_da = xr.DataArray(
        lpips_time_ensemble.numpy(),
        dims=("time", "ensemble"),
        coords={
            "time": time_coords,
            "ensemble": ensemble_coords,
        },
        name="lpips_time_ensemble",
    )

    lpips_channel_da = xr.DataArray(
        lpips_channel.numpy(),
        dims=("channel",),
        coords={"channel": channel_coords},
        name="lpips_channelwise",
    )

    lpips_ensemble_da = xr.DataArray(
        lpips_ensemble.numpy(),
        dims=("ensemble",),
        coords={"ensemble": ensemble_coords},
        name="lpips_ensemblewise",
    )

    lpips_ds = xr.Dataset(
        {
            "lpips_full": lpips_full_da,
            "lpips_time_channel": lpips_time_channel_da,
            "lpips_time_ensemble": lpips_time_ensemble_da,
            "lpips_channelwise": lpips_channel_da,
            "lpips_ensemblewise": lpips_ensemble_da,
        }
    )

    lpips_ds.attrs["lpips_scalar"] = float(lpips_scalar)
    lpips_ds.attrs["normalization"] = "fixed stats_dict output mean/std, clipped by clip_sigma"
    lpips_ds.attrs["shape_order"] = "time, ensemble, channel, y, x"

    return lpips_channel, lpips_ensemble, lpips_scalar, lpips_ds


# -----------------------------
# Saving
# -----------------------------

def save_one_model_lpips(
    title: str,
    file_path: str,
    out_path: Path,
    stats_dict: dict,
    target_ensemble: int = 12,
    clip_sigma: float = 3.0,
    batch_size: int = 256,
    overwrite: bool = False,
):
    print(f"\n----- {title} -----")
    print(f"file: {file_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    scalar_path = out_path / f"{title}.txt"
    channel_path = out_path / f"{title}_channelwise.txt"
    ensemble_path = out_path / f"{title}_ensemblewise.txt"
    nc_path = out_path / f"{title}.nc"

    if (
        not overwrite
        and scalar_path.exists()
        and channel_path.exists()
        and ensemble_path.exists()
        and nc_path.exists()
    ):
        print("All stats exist, moving to next one")
        return

    truth, pred, root, inp = open_samples(file_path)
    pred = pred.load()
    truth = truth.load()

    pred_da, truth_da = dataset_to_ensemble_da(
        pred=pred,
        truth=truth,
        target_ensemble=target_ensemble,
    )

    print("pred_da sizes:", pred_da.sizes)
    print("truth_da sizes:", truth_da.sizes)

    pred_arr, truth_arr = normalize_with_stats(
        pred_da=pred_da,
        truth_da=truth_da,
        stats_dict=stats_dict,
        clip_sigma=clip_sigma,
    )

    lpips_channel, lpips_ensemble, lpips_scalar, lpips_ds = compute_lpips_e12_fixednorm(
        pred_arr=pred_arr,
        truth_arr=truth_arr,
        channel_coords=pred_da.channel.values,
        time_coords=pred_da.time.values,
        ensemble_coords=pred_da.ensemble.values,
        batch_size=batch_size,
    )

    print(f"lpips scalar: {lpips_scalar}")
    print(f"lpips ensemblewise: {lpips_ensemble.numpy()}")

    # Scalar
    if overwrite or not scalar_path.exists():
        with open(scalar_path, "w") as f:
            f.write(f"{lpips_scalar}\n")
    else:
        print("Found scalar txt - skipping save")

    # Channel-wise
    if overwrite or not channel_path.exists():
        with open(channel_path, "w") as f:
            for i, ch in enumerate(pred_da.channel.values):
                f.write(f"{ch}: {lpips_channel[i].item()}\n")
    else:
        print("Found channelwise txt - skipping save")

    # Ensemble-wise
    if overwrite or not ensemble_path.exists():
        with open(ensemble_path, "w") as f:
            for i, ens in enumerate(pred_da.ensemble.values):
                f.write(f"{ens}: {lpips_ensemble[i].item()}\n")
    else:
        print("Found ensemblewise txt - skipping save")

    # NetCDF
    if overwrite or not nc_path.exists():
        if nc_path.exists():
            nc_path.unlink()
        lpips_ds.to_netcdf(nc_path, mode="w", engine="netcdf4")
    else:
        print("Found nc file - skipping save")

    print("saved:")
    print(f"  {scalar_path}")
    print(f"  {channel_path}")
    print(f"  {ensemble_path}")
    print(f"  {nc_path}")
    print("----------------")


def save_lpips_e12_fixednorm(
    out_path: Path,
    stats_dict: dict,
    target_ensemble: int = 12,
    clip_sigma: float = 3.0,
    batch_size: int = 256,
    overwrite: bool = False,
    **kwargs,
):
    """
    Usage:
        save_lpips_e12_fixednorm(
            out_path=out,
            stats_dict=stats_dict,
            unet=[".../600_samples.nc"],
            ...
        )
    """

    out_path.mkdir(parents=True, exist_ok=True)

    for title, data in kwargs.items():
        file_path = data[0]

        try:
            save_one_model_lpips(
                title=title,
                file_path=file_path,
                out_path=out_path,
                stats_dict=stats_dict,
                target_ensemble=target_ensemble,
                clip_sigma=clip_sigma,
                batch_size=batch_size,
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"[ERROR] {title} failed: {repr(e)}")
            raise


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    base_dir = Path("/mnt/hrrr_era5_0528/experiment_result")

    out = Path("/mnt/hrrr_era5_0528/experiment_result/metrics/lpips_e12_fixednorm")
    out.mkdir(parents=True, exist_ok=True)

    stats_path = Path("/mnt/hrrr_era5_0528/hrrr_era5_test_2021_stats.json")

    with open(stats_path, "r") as f:
        stats_dict = json.load(f)

    save_lpips_e12_fixednorm(
        out_path=out,
        stats_dict=stats_dict,
        target_ensemble=12,
        clip_sigma=3.0,
        batch_size=256,
        overwrite=False,

        # unet=[str(base_dir / "unet" / "600_samples.nc")],
        # cdm=[str(base_dir / "cdm" / "600_samples.nc")],
        # corrdiff_m=[str(base_dir / "corrdiff_m" / "600_samples.nc")],
        # corrdiff=[str(base_dir / "corrdiff" / "600_samples.nc")],
        # convfno=[str(base_dir / "convfno" / "600_samples.nc")],
        # swinir=[str(base_dir / "swinir" / "600_samples.nc")],
        # cfg=[str(base_dir / "cfg" / "600_samples.nc")],
        uq_rmse_gt=[str(base_dir / "uq_rmse_0625" / "600_samples.nc")],
        uq_quantiles_gt=[str(base_dir / "uq_quantiles_gt" / "600_samples.nc")],
        # rematch_u=[str(base_dir / "rematch_u" / "600_samples.nc")],
        # rematch_s=[str(base_dir / "rematch_s" / "600_samples.nc")],
    )