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

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_alex.eval()


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out

# Following CorrDiff impelementation
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


def compute_lpips_stats(pred_arr: np.ndarray, truth_arr: np.ndarray, channel_coords, time_coords):
    
    pred_t = torch.tensor(pred_arr)
    truth_t = torch.tensor(truth_arr)
    print(f"PRED_T SHAPE IS {pred_t.shape}, {truth_t.shape}") # shld be pred_t, truth_t → (T, C, H, W) yes!
    T,C,H,W = pred_t.shape

    #normalize images across all channels instead of each image -> this is to preserve magnitude information since a 20m/s diff should be worse than a 2 m/s. but also preserve class information
    # compute per-channel min/max across time+space
    min_val = torch.minimum(
        pred_t.amin(dim=(0, 2, 3), keepdim=True),
        truth_t.amin(dim=(0, 2, 3), keepdim=True),
    )

    max_val = torch.maximum(
        pred_t.amax(dim=(0, 2, 3), keepdim=True),
        truth_t.amax(dim=(0, 2, 3), keepdim=True),
    )

    # pred_norm = 2 * (pred_t - min_val) / (max_val - min_val + 1e-6) - 1
    # truth_norm = 2 * (truth_t - min_val) / (max_val - min_val + 1e-6) - 1
    pred_norm = pred_t
    truth_norm = truth_t

    # we want LPIPS per sample and then average the samples per channel + average everything
    # repeat for RGB dimensions
    pred_flat = pred_norm.permute(1, 0, 2, 3).reshape(C*T, H, W).unsqueeze(1)
    truth_flat = truth_norm.permute(1, 0, 2, 3).reshape(C*T, H, W).unsqueeze(1)
    print(f"flat arr shape {pred_flat.shape}")
    assert (pred_norm[0,0,0,0] == pred_flat[0,0,0,0])  # should match (c0,t0)
    pred_rgb = pred_flat.repeat(1, 3, 1, 1).float()
    truth_rgb = truth_flat.repeat(1, 3, 1, 1).float()
    with torch.no_grad():
        d = loss_fn_alex(pred_rgb, truth_rgb) # outputs T*C,1,1,1
        print(f"res shape {d.shape}")

    d = d.view(C, T)

    lpips_channel = d.mean(dim=1)   # (C,)
    lpips_scalar = d.mean().item()  # scalar

    lpips_da = xr.DataArray(
    d.detach().cpu().numpy(),
    dims=("channel", "time"),
    coords={
        "channel": channel_coords,
        "time": time_coords,
    },
    name="lpips"
)
    lpips_ds = lpips_da.to_dataset(dim="channel")

    return lpips_channel, lpips_scalar, lpips_ds


def save_lpips_stats(out_path: Path, stats_dict, **kwargs):

    for i, (title, data) in enumerate(kwargs.items()):
        print(f"----- {title} --------")
        if (os.path.exists(out_path / "lpips" / f"{title}_channelwise.txt") and os.path.exists(out_path / "lpips" / f"{title}.txt") and os.path.exists(out_path / "lpips" / f"{title}.nc")):
            print(f"All stats exist, moving to next one")
            continue
        file_path = data[0]
        truth, pred, root, inp = open_samples(file_path)
        pred = pred.load()
        print(f"before")
        print(pred)
        pred_mean = pred.mean(dim="ensemble")
        assert list(pred_mean.data_vars) == list(truth.data_vars)
        pred_da = pred_mean.to_array(dim="channel")
        truth_da = truth.to_array(dim="channel")
        pred_da = pred_da.transpose("time", "channel", "y", "x")
        truth_da = truth_da.transpose("time", "channel", "y", "x")
        
        pred_arr = pred_da.values
        truth_arr = truth_da.values


        channels = pred_da.channel.values

        means = np.array([stats_dict["output"][ch]["mean"] for ch in channels])
        stds  = np.array([stats_dict["output"][ch]["std"]  for ch in channels])
        means = means.reshape(1, len(channels), 1, 1)  # (1,C,1,1)
        stds  = stds.reshape(1, len(channels), 1, 1)

        pred_arr = (pred_arr - means) / (stds + 1e-6)
        truth_arr = (truth_arr - means) / (stds + 1e-6)

        #map to -1,1
        pred_arr = np.clip(pred_arr / 3.0, -1.0, 1.0)
        truth_arr = np.clip(truth_arr / 3.0, -1.0, 1.0)

  
        lpips_channel, lpips_scalar, lpips_da = compute_lpips(pred_arr, truth_arr, pred_da.channel.values, pred_da.time.values)

        print(lpips_da)
   
        
        if os.path.exists(out_path / "lpips" / f"{title}_channelwise.txt"):
            print(f"Found channelwise rmse - skipping save")
        else:
            with open(out_path / "lpips" / f"{title}_channelwise.txt", "a") as f:
                for i, var in enumerate(pred_mean.data_vars):
                    f.write(f"{var}: {lpips_channel[i]}\n")

        if os.path.exists(out_path / "lpips" / f"{title}.txt"):
            print(f"Found lpips - skipping save")
        else:
            with open(out_path / "lpips" / f"{title}.txt", "a") as f:
                f.write(f"{lpips_scalar}\n")

        if os.path.exists(out_path / "lpips" / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path / "lpips" / f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            lpips_da.to_netcdf(OUT_NC, mode="w", engine="netcdf4")

        print("---------------")


# out = Path("/data/shared_experiment/metrics/")
# with open("/data/corrdiff3d/hrrrmini_east_test_ble_temporal_stats.json", 'r') as file:
#     stats_dict = json.load(file)

# os.makedirs(out, exist_ok=True)
# save_lpips_stats(
#     out_path=out,
#     stats_dict=stats_dict,
#     #corrdiff=["/data/shared_experiment/baselines/standard_model/2021_600samples_8milreg2018-2020_8milres2018-2020.nc"],
#     ot_no_penalty_regularization=["/data/shared_experiment/optimal_transport/no_penalty_regularization/2021_600samples_8milreg2018-2019_8milres2018-2020.nc"],
#     ot_top_m=["/data/shared_experiment/optimal_transport/top_m/2021_600samples_8milreg2018-2019_8milres2018-2020.nc"],
#     q5q95=["/data/shared_experiment/uncertainty_estimator/scalars/q5andq95/2021_600samples_8milreg2018-2019_8milbias2020w3milreg_8milres2018-2020_resl2loss.nc"],
#     q5q95GTmasked=["/data/shared_experiment/uncertainty_estimator/scalars/q5andq95/2021_600samples_8milreg2018-2019_8milbias2020w3milreg_8milres2018-2020_resl2lossGTmasked.nc"],
#     rmse=["/data/shared_experiment/uncertainty_estimator/scalars/rmse/2021_600samples_8milreg2018-2019_3milbias2020w3milreg_8milres2018-2020_resl2loss.nc"],
#     rmse_GTmasked=["/data/shared_experiment/uncertainty_estimator/scalars/rmse/2021_600samples_gt_rmse_8milreg2018-2019_3milbias2020w3milreg_8milres2018-2020_resl2loss.nc"],
#     rdit=["/data/shared_experiment/rdit/2021_600samples_8milreg2018-2020_8milres2018-2020_multgammas.nc"],
#     tinymodel=["/data/shared_experiment/baselines/tiny_model/2021_600samples_3miltinyreg2018-2020_8milres2018-2020.nc"],
# )


use_latlon=True

def plot_field(ax, C, title, vmin=None, vmax=None, isDiff=False, isUncertainity=False):
    C = np.asarray(C)
    if use_latlon:
        # im = ax.pcolormesh(lon, lat, C, shading="auto", vmin=vmin, vmax=vmax)
        # ax.set_xlabel("lon"); ax.set_ylabel("lat")

        # Hardcoding lats/lons for now - east region
        lon_max, lon_min = -75.20870000000002, -82.55009999999999
        lat_min, lat_max = 36.0795,41.8914

        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                    crs=ccrs.PlateCarree())

        # add background
        # gridlines with labels
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        ax.coastlines('50m', linewidth=0.5, alpha=0.5)
        H, W = C.shape
        lons = np.linspace(lon_min, lon_max, W)
        lats = np.linspace(lat_min, lat_max, H)
        lon2d, lat2d = np.meshgrid(lons, lats)
        xx = np.arange(W + 1)
        yy = np.arange(H + 1)
        im = ax.pcolormesh(lon2d, lat2d, C, shading="auto", vmin=vmin, vmax=vmax,transform=ccrs.PlateCarree())
        if isDiff:
            im = ax.pcolormesh(lon2d, lat2d, C, shading="auto", vmin=vmin, vmax=vmax,transform=ccrs.PlateCarree(),cmap="RdBu_r") # reversed, blue -->red
        elif isUncertainity:
            im = ax.pcolormesh(lon2d, lat2d, C, shading="auto", vmin=vmin, vmax=vmax,transform=ccrs.PlateCarree(),cmap="coolwarm") # default blue --> red i think
        ax.set_xlabel("x"); ax.set_ylabel("y")
    else:
        # fallback: plot in pixel indices (always finite)
        H, W = C.shape
        xx = np.arange(W + 1)
        yy = np.arange(H + 1)
        im = ax.pcolormesh(xx, yy, C, shading="auto", vmin=vmin, vmax=vmax)
        if isDiff:
            im = ax.pcolormesh(xx, yy, C, shading="auto", vmin=vmin, vmax=vmax,cmap="RdBu_r")
        
        ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("\n".join(wrap(f"{title}", 30)))
    return im

def compute_lpips(pred_arr: np.ndarray, truth_arr: np.ndarray, channel_coords, time_coords):
    
    pred_t = torch.tensor(pred_arr)
    truth_t = torch.tensor(truth_arr)
    #print(f"PRED_T SHAPE IS {pred_t.shape}, {truth_t.shape}") # shld be pred_t, truth_t → (T, C, H, W) yes!
    T,C,H,W = pred_t.shape

    #normalize images across all channels instead of each image -> this is to preserve magnitude information since a 20m/s diff should be worse than a 2 m/s. but also preserve class information
    q_low = 0.01
    q_high = 0.99

    # 1. Flatten H and W dimensions
    pred_flat = pred_t.view(T,C, -1)
    truth_flat = truth_t.view(T,C, -1)

    # 2. Compute quantile over the last dimension (H*W)
    # Output shape:  (Batch, Channels)
    pred_qhigh = torch.quantile(pred_flat, q_high, dim=2)
    truth_qhigh = torch.quantile(truth_flat, q_high, dim=2)
    pred_qlow = torch.quantile(pred_flat, q_low, dim=2)
    truth_qlow = torch.quantile(truth_flat, q_low, dim=2)

    plow,_=torch.min(pred_qlow, dim=0)
    tlow,_=torch.min(truth_qlow, dim=0)
    phigh,_=torch.max(pred_qhigh, dim=0)
    thigh,_=torch.max(truth_qhigh, dim=0)

    min_val = torch.minimum(plow, tlow)
    max_val = torch.maximum(phigh, thigh)


    pred_norm = 2 * (pred_t - min_val.view(1,C,1,1)) / (max_val - min_val + 1e-6).view(1,C,1,1) - 1
    truth_norm = 2 * (truth_t - min_val.view(1,C,1,1)) / (max_val - min_val + 1e-6).view(1,C,1,1) - 1
    pred_norm = pred_norm.clamp(-1, 1)
    truth_norm = truth_norm.clamp(-1, 1)
    print(f"pred norm max and min shld be -1,1: {pred_norm.max(), pred_norm.min()}") # <-checked this its good

    # fig, axes = plt.subplots(2,2,
    #         subplot_kw={'projection': ccrs.PlateCarree()},
    #         constrained_layout=True
    #     )
        
    # plt.suptitle(f'(lpips channelwise norm check)')

    # im = plot_field(
    #     axes[0,0], pred_norm[0,0],
    #     f"PRED UWND 50 hPa",
    #     vmin=truth_norm[0,0].min(), vmax=truth_norm[0,0].max()
    # )
    # fig.colorbar(im, ax=axes[0,0], shrink=0.85)
    # im = plot_field(
    #     axes[0,1], truth_norm[0,0],
    #     f"TRUTH UWND 50 hPa",
    #     vmin=truth_norm[0,0].min(), vmax=truth_norm[0,0].max()
    # )
    # fig.colorbar(im, ax=axes[0,1], shrink=0.85)
    # im = plot_field(
    #     axes[1,0], pred_norm[0,1],
    #     f"PRED VWND 50 hPa",
    #     vmin=truth_norm[0,1].min(), vmax=truth_norm[0,1].max())
    # fig.colorbar(im, ax=axes[1,0], shrink=0.85)
    # im = plot_field(
    #     axes[1,1], truth_norm[0,1],
    #     f"TRUTH VWND 50 hPa",
    #     vmin=truth_norm[0,1].min(), vmax=truth_norm[0,1].max()
    # )
    # fig.colorbar(im, ax=axes[1,1], shrink=0.85)
    # plt.savefig("metrics_test.png")

    
    # we want LPIPS per sample and then average the samples per channel + average everything
    # repeat for RGB dimensions
    pred_flat = pred_norm.permute(1, 0, 2, 3).reshape(C*T, H, W).unsqueeze(1)
    truth_flat = truth_norm.permute(1, 0, 2, 3).reshape(C*T, H, W).unsqueeze(1)
    print(f"flat arr shape {pred_flat.shape}")
    assert (pred_norm[0,0,0,0] == pred_flat[0,0,0,0])  # should match (c0,t0)
    pred_rgb = pred_flat.repeat(1, 3, 1, 1)
    truth_rgb = truth_flat.repeat(1, 3, 1, 1)
    with torch.no_grad():
        d = loss_fn_alex(pred_rgb, truth_rgb) # outputs T*C,1,1,1
        print(f"res shape {d.shape}")

    d = d.view(C, T)

    lpips_channel = d.mean(dim=1)   # (C,)
    lpips_scalar = d.mean().item()  # scalar

    lpips_da = xr.DataArray(
    d.detach().cpu().numpy(),
    dims=("channel", "time"),
    coords={
        "channel": channel_coords,
        "time": time_coords,
    },
    name="lpips"
)
    lpips_ds = lpips_da.to_dataset(dim="channel")

    return lpips_channel, lpips_scalar, lpips_ds


def save_lpips(out_path: Path, **kwargs):

    for i, (title, data) in enumerate(kwargs.items()):
        print(f"----- {title} --------")
        if (os.path.exists(out_path / f"{title}_channelwise.txt") and os.path.exists(out_path / f"{title}.txt") and os.path.exists(out_path / f"{title}.nc")):
            print(f"All stats exist, moving to next one")
            continue
        file_path = data[0]
        truth, pred, root, inp = open_samples(file_path)
        pred = pred.load()
        pred_mean = pred.mean(dim="ensemble")
        assert list(pred_mean.data_vars) == list(truth.data_vars)
        pred_da = pred_mean.to_array(dim="channel")
        truth_da = truth.to_array(dim="channel")
        pred_da = pred_da.transpose("time", "channel", "y", "x")
        truth_da = truth_da.transpose("time", "channel", "y", "x")
        pred_arr = pred_da.values
        truth_arr = truth_da.values

        lpips_channel, lpips_scalar, lpips_da = compute_lpips(pred_arr, truth_arr, pred_da.channel.values, pred_da.time.values)

        print(lpips_da)
   
        
        if os.path.exists(out_path  / f"{title}_channelwise.txt"):
            print(f"Found channelwise rmse - skipping save")
        else:
            with open(out_path  / f"{title}_channelwise.txt", "a") as f:
                for i, var in enumerate(pred_mean.data_vars):
                    f.write(f"{var}: {lpips_channel[i]}\n")

        if os.path.exists(out_path / f"{title}.txt"):
            print(f"Found lpips - skipping save")
        else:
            with open(out_path / f"{title}.txt", "a") as f:
                f.write(f"{lpips_scalar}\n")

        if os.path.exists(out_path / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path /  f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            lpips_da.to_netcdf(OUT_NC, mode="w", engine="netcdf4")

        print("---------------")

def compute_lpips_normalizechannel(pred_arr: np.ndarray, truth_arr: np.ndarray, channel_coords, time_coords):
    
    pred_t = torch.tensor(pred_arr)
    truth_t = torch.tensor(truth_arr)
    #print(f"PRED_T SHAPE IS {pred_t.shape}, {truth_t.shape}") # shld be pred_t, truth_t → (T, C, H, W) yes!
    T,E,C,H,W = pred_t.shape

    #normalize images across all channels instead of each image -> this is to preserve magnitude information since a 20m/s diff should be worse than a 2 m/s. but also preserve class information
    q_low = 0.01
    q_high = 0.99

    # 1. Flatten H and W dimensions
    pred_flat = pred_t.reshape(T,C, -1)
    truth_flat = truth_t.reshape(T,C, -1)

    # 2. Compute quantile over the last dimension (H*W)
    # Output shape:  (Times, Channels) -> 600,10
    pred_qhigh = torch.quantile(pred_flat, q_high, dim=2)
    truth_qhigh = torch.quantile(truth_flat, q_high, dim=2)
    pred_qlow = torch.quantile(pred_flat, q_low, dim=2)
    truth_qlow = torch.quantile(truth_flat, q_low, dim=2)

    plow,_=torch.min(pred_qlow, dim=0) # 10
    tlow,_=torch.min(truth_qlow, dim=0)
    phigh,_=torch.max(pred_qhigh, dim=0)
    thigh,_=torch.max(truth_qhigh, dim=0)
  

    min_val = torch.minimum(plow, tlow) # 10
    max_val = torch.maximum(phigh, thigh)


    pred_norm = 2 * (pred_t - min_val.view(1,1,C,1,1)) / (max_val - min_val + 1e-6).view(1,1,C,1,1) - 1
    truth_norm = 2 * (truth_t - min_val.view(1,1,C,1,1)) / (max_val - min_val + 1e-6).view(1,1,C,1,1) - 1
    print(f"pred norm max and min shld be -1,1: {pred_norm.max(), pred_norm.min()}") # <-checked this its good
    pred_norm = pred_norm.clamp(-1, 1) # T,E,C,H,W
    truth_norm = truth_norm.clamp(-1, 1) # T,E,C,H,W
 
    # fig, axes = plt.subplots(2,2,
    #         subplot_kw={'projection': ccrs.PlateCarree()},
    #         constrained_layout=True
    #     )
        
    # plt.suptitle(f'(lpips channelwise norm check)')

    # im = plot_field(
    #     axes[0,0], pred_norm[0,0],
    #     f"PRED UWND 50 hPa",
    #     vmin=truth_norm[0,0].min(), vmax=truth_norm[0,0].max()
    # )
    # fig.colorbar(im, ax=axes[0,0], shrink=0.85)
    # im = plot_field(
    #     axes[0,1], truth_norm[0,0],
    #     f"TRUTH UWND 50 hPa",
    #     vmin=truth_norm[0,0].min(), vmax=truth_norm[0,0].max()
    # )
    # fig.colorbar(im, ax=axes[0,1], shrink=0.85)
    # im = plot_field(
    #     axes[1,0], pred_norm[0,1],
    #     f"PRED VWND 50 hPa",
    #     vmin=truth_norm[0,1].min(), vmax=truth_norm[0,1].max())
    # fig.colorbar(im, ax=axes[1,0], shrink=0.85)
    # im = plot_field(
    #     axes[1,1], truth_norm[0,1],
    #     f"TRUTH VWND 50 hPa",
    #     vmin=truth_norm[0,1].min(), vmax=truth_norm[0,1].max()
    # )
    # fig.colorbar(im, ax=axes[1,1], shrink=0.85)
    # plt.savefig("metrics_test.png")

    
    # we want LPIPS per sample and then average the samples per channel + average everything
    # repeat for RGB dimensions
    pred_flat = pred_norm.permute(0,1,2,3,4).reshape(T*E*C, H, W).unsqueeze(1)
    truth_flat = truth_norm.permute(0,1,2,3,4).reshape(T*E*C, H, W).unsqueeze(1)
    print(f"flat arr shape {pred_flat.shape}")
    assert (pred_norm[0,0,0,0,0] == pred_flat[0,0,0,0])  # should match (c0,t0)
    pred_rgb = pred_flat.repeat(1, 3, 1, 1)
    truth_rgb = truth_flat.repeat(1, 3, 1, 1)
    with torch.no_grad():
        d = loss_fn_alex(pred_rgb, truth_rgb) # outputs T*E*C,1,1,1
        print(f"res shape {d.shape}")

    d = d.view(T,E,C)

    lpips_channel = d.mean(dim=(0,1))   # (C,)
    lpips_scalar = d.mean().item()  # scalar
    lpips_time_channel = d.mean(dim=1)  # average over ensemble → (T, C)

    lpips_da = xr.DataArray(
    lpips_time_channel.cpu().numpy(),
    dims=("time", "channel"),
    coords={
        "time": time_coords,
        "channel": channel_coords,
        
    },
    name="lpips"
)
    lpips_ds = lpips_da.to_dataset(dim="channel")

    return lpips_channel, lpips_scalar, lpips_ds



def compute_lpips_normalizeensembleandchannel(pred_arr: np.ndarray, truth_arr: np.ndarray, channel_coords, time_coords):
    
    pred_t = torch.tensor(pred_arr)
    truth_t = torch.tensor(truth_arr)
    #print(f"PRED_T SHAPE IS {pred_t.shape}, {truth_t.shape}") # shld be pred_t, truth_t → (T, C, H, W) yes!
    T,E,C,H,W = pred_t.shape

    #normalize images across all channels instead of each image -> this is to preserve magnitude information since a 20m/s diff should be worse than a 2 m/s. but also preserve class information
    q_low = 0.01
    q_high = 0.99

    # 1. Flatten H and W dimensions
    pred_flat = pred_t.view(T,E,C, -1)
    truth_flat = truth_t.view(T,E,C, -1)

    # 2. Compute quantile over the last dimension (H*W)
    # Output shape:  (Times, Ensembles, Channels) -> 600,12,10
    pred_qhigh = torch.quantile(pred_flat, q_high, dim=3)
    truth_qhigh = torch.quantile(truth_flat, q_high, dim=3)
    pred_qlow = torch.quantile(pred_flat, q_low, dim=3)
    truth_qlow = torch.quantile(truth_flat, q_low, dim=3)

    plow,_=torch.min(pred_qlow, dim=0) # 12,10
    tlow,_=torch.min(truth_qlow, dim=0)
    phigh,_=torch.max(pred_qhigh, dim=0)
    thigh,_=torch.max(truth_qhigh, dim=0)
  

    min_val = torch.minimum(plow, tlow) # 12,10
    max_val = torch.maximum(phigh, thigh)


    pred_norm = 2 * (pred_t - min_val.view(1,E,C,1,1)) / (max_val - min_val + 1e-6).view(1,E,C,1,1) - 1
    truth_norm = 2 * (truth_t - min_val.view(1,E,C,1,1)) / (max_val - min_val + 1e-6).view(1,E,C,1,1) - 1
    print(f"pred norm max and min shld be -1,1: {pred_norm.max(), pred_norm.min()}") # <-checked this its good
    pred_norm = pred_norm.clamp(-1, 1) # T,E,C,H,W
    truth_norm = truth_norm.clamp(-1, 1) # T,E,C,H,W
 
    # fig, axes = plt.subplots(2,2,
    #         subplot_kw={'projection': ccrs.PlateCarree()},
    #         constrained_layout=True
    #     )
        
    # plt.suptitle(f'(lpips channelwise norm check)')

    # im = plot_field(
    #     axes[0,0], pred_norm[0,0],
    #     f"PRED UWND 50 hPa",
    #     vmin=truth_norm[0,0].min(), vmax=truth_norm[0,0].max()
    # )
    # fig.colorbar(im, ax=axes[0,0], shrink=0.85)
    # im = plot_field(
    #     axes[0,1], truth_norm[0,0],
    #     f"TRUTH UWND 50 hPa",
    #     vmin=truth_norm[0,0].min(), vmax=truth_norm[0,0].max()
    # )
    # fig.colorbar(im, ax=axes[0,1], shrink=0.85)
    # im = plot_field(
    #     axes[1,0], pred_norm[0,1],
    #     f"PRED VWND 50 hPa",
    #     vmin=truth_norm[0,1].min(), vmax=truth_norm[0,1].max())
    # fig.colorbar(im, ax=axes[1,0], shrink=0.85)
    # im = plot_field(
    #     axes[1,1], truth_norm[0,1],
    #     f"TRUTH VWND 50 hPa",
    #     vmin=truth_norm[0,1].min(), vmax=truth_norm[0,1].max()
    # )
    # fig.colorbar(im, ax=axes[1,1], shrink=0.85)
    # plt.savefig("metrics_test.png")

    
    # we want LPIPS per sample and then average the samples per channel + average everything
    # repeat for RGB dimensions
    pred_flat = pred_norm.permute(0,1,2,3,4).reshape(T*E*C, H, W).unsqueeze(1)
    truth_flat = truth_norm.permute(0,1,2,3,4).reshape(T*E*C, H, W).unsqueeze(1)
    print(f"flat arr shape {pred_flat.shape}")
    assert (pred_norm[0,0,0,0,0] == pred_flat[0,0,0,0])  # should match (c0,t0)
    pred_rgb = pred_flat.repeat(1, 3, 1, 1)
    truth_rgb = truth_flat.repeat(1, 3, 1, 1)
    with torch.no_grad():
        d = loss_fn_alex(pred_rgb, truth_rgb) # outputs T*E*C,1,1,1
        print(f"res shape {d.shape}")

    d = d.view(T,E,C)

    lpips_channel = d.mean(dim=(0,1))   # (C,)
    lpips_scalar = d.mean().item()  # scalar
    lpips_time_channel = d.mean(dim=1)  # average over ensemble → (T, C)

    lpips_da = xr.DataArray(
    lpips_time_channel.cpu().numpy(),
    dims=("time", "channel"),
    coords={
        "time": time_coords,
        "channel": channel_coords,
        
    },
    name="lpips"
)
    lpips_ds = lpips_da.to_dataset(dim="channel")

    return lpips_channel, lpips_scalar, lpips_ds


def save_lpips_ensemble(out_path: Path, **kwargs):

    for i, (title, data) in enumerate(kwargs.items()):
        print(f"----- {title} --------")
        if (os.path.exists(out_path / f"{title}_channelwise.txt") and os.path.exists(out_path / f"{title}.txt") and os.path.exists(out_path / f"{title}.nc")):
            print(f"All stats exist, moving to next one")
            continue
        file_path = data[0]
        truth, pred, root, inp = open_samples(file_path)
        pred = pred.load()
        pred_da = pred.to_array(dim="channel")
        truth_da = truth.to_array(dim="channel")
        pred_da = pred_da.transpose("time", "ensemble", "channel", "y", "x")
        truth_da = truth_da.expand_dims({"ensemble": pred_da.ensemble})
        truth_da = truth_da.transpose("time", "ensemble", "channel", "y", "x")
        pred_arr = pred_da.values
        truth_arr = truth_da.values

        lpips_channel, lpips_scalar, lpips_da = compute_lpips_normalizechannel(pred_arr, truth_arr, pred_da.channel.values, pred_da.time.values)

        print(lpips_da)
   
        
        if os.path.exists(out_path  / f"{title}_channelwise.txt"):
            print(f"Found channelwise rmse - skipping save")
        else:
            with open(out_path  / f"{title}_channelwise.txt", "a") as f:
                for i, var in enumerate(pred.data_vars):
                    f.write(f"{var}: {lpips_channel[i]}\n")

        if os.path.exists(out_path / f"{title}.txt"):
            print(f"Found lpips - skipping save")
        else:
            with open(out_path / f"{title}.txt", "a") as f:
                f.write(f"{lpips_scalar}\n")

        if os.path.exists(out_path / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path /  f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            lpips_da.to_netcdf(OUT_NC, mode="w", engine="netcdf4")

        print("---------------")


out = Path("/data/hrrr_era5_0528/experiment_result/metrics/lpips_channelnorm")
os.makedirs(out, exist_ok=True)
base_dir = Path("/data/hrrr_era5_0528/experiment_result")
save_lpips_ensemble(
    out_path=out,
    cfg=[str(base_dir / "cfg" / "600_samples.nc")],
    convfno=[str(base_dir / "convfno" / "600_samples.nc")],
    corrdiff=[str(base_dir / "corrdiff" / "600_samples.nc")],
    corrdiff_m=[str(base_dir / "corrdiff_m" / "600_samples.nc")],
    rematch_s=[str(base_dir / "rematch_s" / "600_samples.nc")],
    rematch_s_12=[str(base_dir / "rematch_s_12" / "600_samples.nc")],
    rematch_u=[str(base_dir / "rematch_u" / "600_samples.nc")],
    swinir=[str(base_dir / "swinir" / "600_samples.nc")],
    unet=[str(base_dir / "unet" / "600_samples.nc")],
    uq_quantiles=[str(base_dir / "uq_quantiles" / "600_samples.nc")],
    uq_rmse=[str(base_dir / "uq_rmse" / "600_samples.nc")],
)