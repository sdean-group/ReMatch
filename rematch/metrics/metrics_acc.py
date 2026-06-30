import numpy as np
from pathlib import Path
import xarray as xr
import os
import lpips
import torch
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from textwrap import wrap
from math import floor,ceil
import pandas as pd

def open_input_output(f="/mnt/hrrr_era5_0528/hrrr_era5_train_2018_2020.nc"):
    root = xr.open_dataset(f)
    truth = xr.open_dataset(f, group="output")
    inp = xr.open_dataset(f, group="input")
    truth = truth.merge(root)
    inp = inp.merge(root)

    return truth, inp

def compute_acc(pred_arr, truth_arr, clim_arr, channel_coords, time_coords):
    """
    pred_arr: (T,C,H,W)
    truth_arr: (T,C,H,W)
    clim_arr: (T,C,H,W)
    """
    print('computing acc')

    pred_t = torch.tensor(pred_arr)
    truth_t = torch.tensor(truth_arr)
    clim_t = torch.tensor(clim_arr)

    # anomalies
    f = pred_t - clim_t
    y = truth_t - clim_t

    # latitude weights
    H = f.shape[2]
    print(f"H IS {H}")

    # assume evenly spaced latitudes → approximate from [-90,90]
    lats = torch.linspace(36.0795, 41.8914, H)
    w = torch.cos(torch.deg2rad(lats))
    w = w / w.sum()
    w = w.view(1, 1, H, 1)

    num = w*f*y
    num = num.flatten(2) # T,C,S
    num = num.sum(dim=2)

    # denominator
    den = torch.sqrt((w*(f**2)).flatten(2).sum(dim=2) * (w*(y**2)).flatten(2).sum(dim=2))

    acc = num / (den + 1e-8)  # (T,C)
    print(f'acc shpae {acc.shape}')

    acc_channel = acc.mean(dim=0)   # (C,)
    acc_scalar = acc.mean().item()

    acc_da = xr.DataArray(
    acc.cpu().numpy(),
    dims=("time", "channel"),
    coords={
        "time": time_coords,
        "channel": channel_coords,
        
    },
    name="acc"
)
    acc_da = acc_da.to_dataset(dim="channel")

    return acc_channel, acc_scalar, acc_da

def open_samples(f):
    """
    Open prediction and truth samples from a dataset file.

    Parameters:
        f: Path to the dataset file.

    Returns:
        tuple: A tuple containing truth, prediction, and root datasets.
    """
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

#def compute_acc_perchannel(pred_arr: np.ndarray, truth_arr: np.ndarray, channel_coords, time_coords):
    

def save_acc(out_path: Path, **kwargs):
    if os.path.exists(out_path / f"climatology.nc"):
        print(f"Found climatology NC file - skipping build")
        clim = xr.open_dataset(out_path / f"climatology.nc")
        print('done opening file')
        print(clim)
    else:
        climatology_out, climatology_inp = open_input_output()
        #training climatology
        print("-----clim_out----------")
        print(climatology_out)

        clim_ds= climatology_out.assign_coords(
        time=("sample", pd.to_datetime(climatology_out["time"].values))
        )
        print("----add time as coord---")
        print(clim_ds)
        clim_monthly = clim_ds.assign_coords(
            dayofyear=("sample",clim_ds.time.dt.dayofyear.data),
            hour=("sample",clim_ds.time.dt.hour.data)
        )
        print("----add day of year and hour as coord----")
        print(clim_monthly)
        clim_monthly = clim_monthly.drop_vars(['coord'])
        clim = clim_monthly.groupby(["dayofyear", "hour"]).mean("sample")
        print('-------group and average by times----------')
        print(clim)
        OUT_NC = out_path /  f"climatology.nc"
        OUT_NC.parent.mkdir(parents=True, exist_ok=True)
        if OUT_NC.exists():
            OUT_NC.unlink()
        
        clim.to_netcdf(OUT_NC, mode="w", engine="netcdf4")

    for i, (title, data) in enumerate(kwargs.items()):
        print(f"----- {title} --------")
        if (os.path.exists(out_path  / f"{title}_channelwise.txt") and os.path.exists(out_path / f"{title}.txt") and os.path.exists(out_path / f"{title}.nc") and os.path.exists(out_path / f"climatology.nc")):
            print(f"All stats exist, moving to next one")
            continue
        file_path = data[0]
        truth, pred, root, inp = open_samples(file_path)
        nT = pred.sizes["time"]
        index = np.arange(nT)
        time = root["time"].isel(time=index)
        # verified this works
        date_times = pd.to_datetime(root["time"].isel(time=index).values)
        doy = date_times.dayofyear
        hr = date_times.hour
        pred = pred.load()
        pred_mean = pred.mean(dim="ensemble")
        pred_mean = pred_mean.assign_coords(time=time)
        truth = truth.assign_coords(time=time)
        assert list(pred_mean.data_vars) == list(truth.data_vars)
        pred_da = pred_mean.to_array(dim="channel")
        truth_da = truth.to_array(dim="channel")
        pred_da = pred_da.transpose("time", "channel", "y", "x")
        truth_da = truth_da.transpose("time", "channel", "y", "x")
        pred_arr = pred_da.values
        truth_arr = truth_da.values

        # print('starting big loop')

        doy = doy.astype(int) - 1   # 0-based index
        hr = hr.astype(int)

        clim_arr_list = []

        for var in pred_da.channel.values:
            clim_field = clim[var].values  # (y, x, doy, hour)
            clim_sel = clim_field[:, :, doy, hr]   # (y, x, T)

            clim_sel = np.transpose(clim_sel, (2, 0, 1))  # (T, y, x)

            clim_arr_list.append(clim_sel)

        clim_arr = np.stack(clim_arr_list, axis=1)  # (T, C, H, W)
        # print(f"clim arr shape")
        # print(clim_arr.shape)

        acc_channel, acc_scalar, acc_ds = compute_acc(pred_arr, truth_arr, clim_arr, pred_da.channel.values, pred_da.time.values)
        print(acc_scalar)
   
        
        if os.path.exists(out_path  / f"{title}_channelwise.txt"):
            print(f"Found channelwise acc - skipping save")
        else:
            with open(out_path  / f"{title}_channelwise.txt", "a") as f:
                for i, var in enumerate(pred.data_vars):
                    f.write(f"{var}: {acc_channel[i]}\n")

        if os.path.exists(out_path / f"{title}.txt"):
            print(f"Found lpips - skipping save")
        else:
            with open(out_path / f"{title}.txt", "a") as f:
                f.write(f"{acc_scalar}\n")

        if os.path.exists(out_path / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path /  f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            acc_ds.to_netcdf(OUT_NC, mode="w", engine="netcdf4")

        print("---------------")


out = Path("/mnt/hrrr_era5_0528/experiment_result/metrics_paper/acc_hrrrclimatology")
os.makedirs(out, exist_ok=True)

base_dir = Path("/mnt/hrrr_era5_0528/experiment_result")


save_acc(
    out_path=out,
    # unet=[str(base_dir / "unet" / "600_samples.nc")],
    # cdm=[str(base_dir / "cdm" / "600_samples.nc")],
    # corrdiff_m=[str(base_dir / "corrdiff_m" / "600_samples.nc")],
    # corrdiff=[str(base_dir / "corrdiff" / "600_samples.nc")],
    # convfno=[str(base_dir / "convfno" / "600_samples.nc")],
    # swinir=[str(base_dir / "swinir" / "600_samples.nc")],
    # cfg=[str(base_dir / "cfg" / "600_samples.nc")],
    # # uq_rmse=[str(base_dir / "uq_rmse" / "600_samples.nc")],
    # # uq_quantiles=[str(base_dir / "uq_quantiles" / "600_samples.nc")],
    # uq_rmse_gt=[str(base_dir / "uq_rmse_0625" / "600_samples.nc")],
    # uq_quantiles_gt=[str(base_dir / "uq_quantiles_gt" / "600_samples.nc")],
    # rematch_u=[str(base_dir / "rematch_u" / "600_samples.nc")],
    rematch_s_correct=[str(base_dir / "rematch_s" / "rematchs_again_600_samples.nc")],
)