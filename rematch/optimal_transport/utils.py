
import torch
import numpy as np
import xarray as xr
from typing import Dict, Optional
from pathlib import Path
from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from textwrap import wrap
import os

def ds_to_array(ds: xr.Dataset, name: str = "variable") -> xr.DataArray:
    if len(ds.data_vars) == 0:
        raise ValueError("Dataset has no data variables.")
    return ds.to_array(dim=name)


def _ensure_parent(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _remove_if_exists(path: str | Path):
    path = Path(path)
    if path.exists():
        path.unlink()

def _attach_missing_coords_from_ref(
    ds: xr.Dataset,
    ref_ds: xr.Dataset,
    required_dims=("time", "y", "x"),
) -> xr.Dataset:
    """
    If ds is missing coordinates for dims like time/y/x, copy them from ref_ds.
    If ref_ds also lacks them, create simple integer coordinates.
    """
    coords_to_add = {}

    for dim in required_dims:
        if dim in ds.dims and dim not in ds.coords:
            if dim in ref_ds.coords and ref_ds.sizes.get(dim) == ds.sizes.get(dim):
                coords_to_add[dim] = ref_ds.coords[dim].values
            else:
                coords_to_add[dim] = np.arange(ds.sizes[dim])

    if coords_to_add:
        ds = ds.assign_coords(coords_to_add)

    return ds
def _build_source_only_root_ds(source_root_ds: xr.Dataset, source_truth_ds: xr.Dataset) -> xr.Dataset:
    """
    Keep target root structure, but make sure time coord length matches target truth.
    """
    root = source_root_ds.copy()

    T = source_truth_ds.sizes["time"]

    # time coord sync
    if "time" in source_root_ds.coords and len(source_root_ds.coords["time"]) == T:
        time_vals = source_root_ds.coords["time"].values
    else:
        time_vals = source_truth_ds.coords["time"].values if "time" in source_truth_ds.coords else np.arange(T)

    root = root.assign_coords(time=time_vals)

    return root


def _build_merged_root_ds(
    source_root_ds: xr.Dataset,
    target_root_ds: xr.Dataset,
    source_truth_ds: xr.Dataset,
    target_truth_ds: xr.Dataset,
) -> xr.Dataset:
    """
    Build merged root dataset:
      - concatenate time from source root + target root
      - keep static vars like lat/lon from target if available
      - preserve attrs from target and record source path info later if desired
    """
    Tt = target_truth_ds.sizes["time"]
    Ts = source_truth_ds.sizes["time"]

    # time values
    if "time" in source_root_ds.coords and len(source_root_ds.coords["time"]) == Ts:
        t_source = source_root_ds.coords["time"].values
    elif "time" in target_truth_ds.coords:
        t_source = source_truth_ds.coords["time"].values
    else:
        t_source = np.arange(Ts)

    if "time" in target_root_ds.coords and len(target_root_ds.coords["time"]) == Tt:
        t_target = target_root_ds.coords["time"].values
    elif "time" in source_truth_ds.coords:
        t_target = target_truth_ds.coords["time"].values
    else:
        t_target = np.arange(Tt)

    merged_time = np.concatenate([t_source, t_target], axis=0)

    # static vars from target root first
    data_vars = {}
    coords = {"time": merged_time}

    # preserve y/x coords if present, otherwise use dims from target truth
    if "y" in source_root_ds.coords:
        coords["y"] = source_root_ds.coords["y"].values
    elif "y" in target_truth_ds.coords:
        coords["y"] = source_truth_ds.coords["y"].values
    else:
        coords["y"] = np.arange(source_truth_ds.sizes["y"])

    if "x" in source_root_ds.coords:
        coords["x"] = source_root_ds.coords["x"].values
    elif "x" in source_truth_ds.coords:
        coords["x"] = source_truth_ds.coords["x"].values
    else:
        coords["x"] = np.arange(source_truth_ds.sizes["x"])

    # copy root data vars that are not time-dependent from target first
    for name, da in source_root_ds.data_vars.items():
        data_vars[name] = da

    # add any missing static vars from source
    for name, da in target_root_ds.data_vars.items():
        if name not in data_vars:
            data_vars[name] = da

    merged_root = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(source_root_ds.attrs))

    # optionally preserve source attrs too
    for k, v in target_root_ds.attrs.items():
        if k not in merged_root.attrs:
            merged_root.attrs[k] = v

    return merged_root
def _maybe_squeeze_prediction(ds: xr.Dataset) -> xr.Dataset:
    """
    If prediction dataset has ensemble dimension of size 1, drop it.
    """
    if "ensemble" in ds.dims:
        # if ds.sizes["ensemble"] != 1:
        #     raise ValueError(
        #         f"Prediction dataset has ensemble dim with size {ds.sizes['ensemble']}, expected 1."
        #     )
        ds = ds.isel(ensemble=0, drop=True)
    return ds

def _array_to_dataset_like(
    arr_c_thw: np.ndarray,   # (C, T, H, W)
    ref_ds: xr.Dataset,
    channel_names: List[str],
) -> xr.Dataset:
    if arr_c_thw.ndim != 4:
        raise ValueError(f"Expected arr shape (C,T,H,W), got {arr_c_thw.shape}")

    C, T, H, W = arr_c_thw.shape
    if len(channel_names) != C:
        raise ValueError(
            f"channel_names length {len(channel_names)} does not match C={C}"
        )

    coords = {}
    for dim, size in [("time", T), ("y", H), ("x", W)]:
        if dim in ref_ds.coords and ref_ds.sizes.get(dim) == size:
            coords[dim] = ref_ds.coords[dim].values
        else:
            coords[dim] = np.arange(size)

    data_vars = {}
    for c, name in enumerate(channel_names):
        data_vars[name] = (("time", "y", "x"), arr_c_thw[c].astype(np.float32))

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _make_encoding(ds: xr.Dataset, compress_level: int = 4):
    encoding = {}
    for var_name in ds.data_vars:
        var = ds[var_name]
        shape = var.shape
        # expected (time, y, x)
        if len(shape) == 3:
            T, H, W = shape
            encoding[var_name] = {
                "zlib": True,
                "complevel": compress_level,
                "chunksizes": (min(T, 256), min(H, 64), min(W, 64)),
            }
        else:
            encoding[var_name] = {"zlib": True, "complevel": compress_level}
    return encoding


def _save_grouped_nc(
    save_path: str | Path,
    root_ds: xr.Dataset,
    truth_ds: xr.Dataset,
    pred_ds: xr.Dataset,
    inp_ds: xr.Dataset,
    compress_level: int = 4,
):
    save_path = _ensure_parent(save_path)
    _remove_if_exists(save_path)

    # 1. save root dataset first
    root_ds.to_netcdf(
        save_path,
        mode="w",
        engine="netcdf4",
        encoding=_make_encoding(root_ds, compress_level) if len(root_ds.data_vars) > 0 else None,
    )

    # 2. append groups
    truth_ds.to_netcdf(
        save_path,
        mode="a",
        group="truth",
        engine="netcdf4",
        encoding=_make_encoding(truth_ds, compress_level),
    )
    pred_ds.to_netcdf(
        save_path,
        mode="a",
        group="prediction",
        engine="netcdf4",
        encoding=_make_encoding(pred_ds, compress_level),
    )
    inp_ds.to_netcdf(
        save_path,
        mode="a",
        group="input",
        engine="netcdf4",
        encoding=_make_encoding(inp_ds, compress_level),
    )

def save_ot_dataset_as_nc(
    source_nc_path: str,
    target_nc_path: str,
    save_ot_reg_path: str,
    save_ot_all_reg_path: str,
    x_res_ot: np.ndarray,
    channel_names: List[str],
    compress_level: int = 4,
):
    # --------------------------------------------------------
    # Load source file (root + groups)
    # --------------------------------------------------------
    source_root_ds = xr.open_dataset(source_nc_path).load()
    source_truth_ds = xr.open_dataset(source_nc_path, group="truth").load()
    source_pred_ds = xr.open_dataset(source_nc_path, group="prediction").load()
    source_inp_ds = xr.open_dataset(source_nc_path, group="input").load()

    source_pred_ds = _maybe_squeeze_prediction(source_pred_ds)

    source_gt_arr = np.array(ds_to_array(source_truth_ds))   # (C, T_x, H, W)

    if source_gt_arr.shape != x_res_ot.shape:
        # raise ValueError(
        #     f"x_res_ot shape {x_res_ot.shape} does not match source truth shape {source_gt_arr.shape}"
        # )
        print(f"x_res_ot shape: {x_res_ot.shape}, source_gt_arr shape: {source_gt_arr.shape}")
        x_res_ot = x_res_ot.reshape(source_gt_arr.shape)

    source_pred_new_arr = source_gt_arr - x_res_ot
    source_pred_new_ds = _array_to_dataset_like(
        source_pred_new_arr,
        ref_ds=source_truth_ds,
        channel_names=channel_names,
    )

    source_root_new_ds = _build_source_only_root_ds(source_root_ds, source_truth_ds)

    # save target-only modified file
    _save_grouped_nc(
        save_path=save_ot_reg_path,
        root_ds=source_root_new_ds,
        truth_ds=source_truth_ds,
        pred_ds=source_pred_new_ds,
        inp_ds=source_inp_ds,
        compress_level=compress_level,
    )
    print(f"Saved target-modified file to {save_ot_reg_path}")

    # --------------------------------------------------------
    # Load target file (root + groups)
    # --------------------------------------------------------
    targe_root_ds = xr.open_dataset(target_nc_path).load()
    target_truth_ds = xr.open_dataset(target_nc_path, group="truth").load()
    target_pred_ds = xr.open_dataset(target_nc_path, group="prediction").load()
    target_inp_ds = xr.open_dataset(target_nc_path, group="input").load()

    target_pred_ds = _maybe_squeeze_prediction(target_pred_ds)
    target_pred_ds = _attach_missing_coords_from_ref(target_pred_ds, target_truth_ds)
    target_inp_ds = _attach_missing_coords_from_ref(target_inp_ds, target_truth_ds)

    # --------------------------------------------------------
    # Reload modified source file
    # --------------------------------------------------------
    mod_root_ds = xr.open_dataset(save_ot_reg_path).load()
    mod_truth_ds = xr.open_dataset(save_ot_reg_path, group="truth").load()
    mod_pred_ds = xr.open_dataset(save_ot_reg_path, group="prediction").load()
    mod_inp_ds = xr.open_dataset(save_ot_reg_path, group="input").load()

    mod_pred_ds = _attach_missing_coords_from_ref(mod_pred_ds, mod_truth_ds)
    mod_inp_ds = _attach_missing_coords_from_ref(mod_inp_ds, mod_truth_ds)

    # --------------------------------------------------------
    # Concatenate groups
    # --------------------------------------------------------
    truth_all_ds = xr.concat(
        [mod_truth_ds, target_truth_ds],
        dim="time",
        coords="minimal",
        compat="override",
        join="override",
    )
    pred_all_ds = xr.concat(
        [mod_pred_ds, target_pred_ds],
        dim="time",
        coords="minimal",
        compat="override",
        join="override",
    )
    inp_all_ds = xr.concat(
        [mod_inp_ds, target_inp_ds],
        dim="time",
        coords="minimal",
        compat="override",
        join="override",
    )

    # --------------------------------------------------------
    # Build merged root dataset
    # --------------------------------------------------------
    merged_root_ds = _build_merged_root_ds(
        source_root_ds=mod_root_ds,
        target_root_ds=targe_root_ds,
        source_truth_ds=mod_truth_ds,
        target_truth_ds=target_truth_ds,
    )

    print("[Merged source_modified + target]")
    print("root merged:", merged_root_ds)
    print("truth merged shape:", np.array(ds_to_array(truth_all_ds)).shape)
    print("pred merged shape:", np.array(ds_to_array(pred_all_ds)).shape)
    print("input merged shape:", np.array(ds_to_array(inp_all_ds)).shape)

    # --------------------------------------------------------
    # Save merged file with root + groups
    # --------------------------------------------------------
    _save_grouped_nc(
        save_path=save_ot_all_reg_path,
        root_ds=merged_root_ds,
        truth_ds=truth_all_ds,
        pred_ds=pred_all_ds,
        inp_ds=inp_all_ds,
        compress_level=compress_level,
    )
    print(f"Saved merged file to {save_ot_all_reg_path}")

def load_training_data(nc_path: str, max_time_idx: int = None):
    """
    Load data from NetCDF file.
    NC structure (confirmed):
        truth/regression : per-var shape (time, y, x)
    After ds_to_array:
        reg_arr : (C, T, H, W)
        res_arr : (C, T, H, W)   
        gt_arr  : (C, T, H, W)
    We iterate over T and build (1,C,H,W) reg and (S,C,H,W) res per sample.
    """
    if max_time_idx is not None:
        seed = 1234
        tmp = xr.open_dataset(nc_path, group="truth")
        t_dim = "time"  
        T_total = tmp.sizes[t_dim]
        tmp.close()
        rng = np.random.default_rng(seed)
        idxs = np.sort(rng.choice(T_total, size=max_time_idx, replace=False))
        truth_ds = xr.open_dataset(nc_path, group="truth").isel({t_dim: idxs})
        reg_ds   = xr.open_dataset(nc_path, group="prediction").isel({t_dim: idxs})

    else:
        truth_ds = xr.open_dataset(nc_path, group="truth")
        # reg_ds   = xr.open_dataset(nc_path, group="prediction").squeeze("ensemble", drop=True)
        reg_ds = xr.open_dataset(nc_path, group="prediction").isel(ensemble=0, drop=True)
        inp_ds = xr.open_dataset(nc_path, group="input")

    res_ds   = truth_ds - reg_ds   # broadcasts reg (T,H,W) over ensemble dim
    
    gt_arr  = ds_to_array(truth_ds)  # (C, T, H, W)
    reg_arr = ds_to_array(reg_ds)    # (C, T, H, W)
    gt_res_arr = ds_to_array(res_ds)    # (C, T, H, W)
    inp_arr = ds_to_array(inp_ds)    # (C, T, H, W)

    T = reg_arr.shape[1]


    print(f"Loaded {T} time steps from {nc_path}")
    print(f"reg shape: {reg_arr.shape}")
    print(f"gt_res shape: {gt_res_arr.shape}")
    print(f"gt shape: {gt_arr.shape}")
    return np.array(inp_arr), np.array(gt_res_arr), np.array(gt_arr)

def load_training_data2(nc_path: str, max_time_idx: int = None):
    """
    Load data from NetCDF file.
    NC structure (confirmed):
        truth/regression : per-var shape (time, y, x)
    After ds_to_array:
        reg_arr : (C, T, H, W)
        res_arr : (C, T, H, W)   
        gt_arr  : (C, T, H, W)
    We iterate over T and build (1,C,H,W) reg and (S,C,H,W) res per sample.
    """
    if max_time_idx is not None:
        seed = 1234
        tmp = xr.open_dataset(nc_path, group="truth")
        t_dim = "time"  
        T_total = tmp.sizes[t_dim]
        tmp.close()
        rng = np.random.default_rng(seed)
        idxs = np.sort(rng.choice(T_total, size=max_time_idx, replace=False))
        truth_ds = xr.open_dataset(nc_path, group="truth").isel({t_dim: idxs})
        pred_ds = xr.open_dataset(nc_path, group="prediction").isel({t_dim: idxs})

    else:
        truth_ds = xr.open_dataset(nc_path, group="truth")
        pred_ds = xr.open_dataset(nc_path, group="prediction")
    
    gt_arr  = ds_to_array(truth_ds)  # (C, T, H, W)
    pred_arr = ds_to_array(pred_ds)  # (S, C, T, H, W)
    T = gt_arr.shape[1]

    print(f"Loaded {T} time steps from {nc_path}")
    return np.array(pred_arr), np.array(gt_arr)

def load_validation_data(nc_path: str, max_time_idx: int = None, return_all: bool = False):
    """
    Load validation data from NetCDF file.

    Target output shapes:
        gt_res_np : (C, T, H, W)
        res_np    : (S, C, T, H, W)

    Assumed dataset structure:
        truth/regression : per-var shape (time, y, x)
        prediction       : per-var shape (ensemble, time, y, x)
    """
    if max_time_idx is not None:
        t_dim = "time"
        idxs = np.arange(max_time_idx)

        gt_ds   = xr.open_dataset(nc_path, group="truth").isel({t_dim: idxs})
        reg_ds  = xr.open_dataset(nc_path, group="regression").isel({t_dim: idxs})
        pred_ds = xr.open_dataset(nc_path, group="prediction").isel({t_dim: idxs})

        inp = xr.open_dataset(nc_path, group="input").isel({t_dim: idxs})
    else:
        gt_ds   = xr.open_dataset(nc_path, group="truth")
        reg_ds  = xr.open_dataset(nc_path, group="regression")
        pred_ds = xr.open_dataset(nc_path, group="prediction")
        inp = xr.open_dataset(nc_path, group="input")
    # prediction has ensemble, regression does not.
    # xarray will broadcast regression over ensemble automatically.
    res_ds = pred_ds - reg_ds

    # truth/regression -> (C, T, H, W)
    gt_arr = ds_to_array(gt_ds)
    reg_arr = ds_to_array(reg_ds)
    gt_res_arr = gt_arr - reg_arr

    gt_np = np.array(gt_arr)            # (C, T, H, W)
    reg_np = np.array(reg_arr)          # (C, T, H, W)
    gt_res_np = np.array(gt_res_arr)    # (C, T, H, W)

    # prediction/residual expected from ds_to_array:
    #   (C, S, T, H, W)
    pred_arr = ds_to_array(pred_ds)
    res_arr  = ds_to_array(res_ds)

    pred_np = np.array(pred_arr).transpose(1, 0, 2, 3, 4)   # -> (S, C, T, H, W)
    res_np  = np.array(res_arr).transpose(1, 0, 2, 3, 4)    # -> (S, C, T, H, W)

    C, T, H, W = gt_res_np.shape

    # print(f"max_time_idx: {max_time_idx}")
    print(f"Loaded {T} time steps from {nc_path}")
    # print(f"gt shape:      {gt_np.shape}")       # (C, T, H, W)
    # print(f"reg shape:     {reg_np.shape}")      # (C, T, H, W)
    # print(f"gt_res shape:  {gt_res_np.shape}")   # (C, T, H, W)
    # print(f"pred shape:    {pred_np.shape}")     # (S, C, T, H, W)
    # print(f"res shape:     {res_np.shape}")      # (S, C, T, H, W)

    if return_all:
        inp_np = np.array(ds_to_array(inp))
        inp_np = inp_np[:10]
        return res_np, gt_res_np, gt_np, reg_np, pred_np, inp_np
    return res_np, gt_res_np
def plot_minmax(*xs):
    pair = np.concatenate([np.asarray(x).ravel() for x in xs])
    vmin, vmax = np.percentile(pair, [0, 100.0])
    return vmin, vmax
# def plot_field(ax, C, title, vmin=None, vmax=None):
#     C = np.asarray(C)
#     H, W = C.shape
#     xx = np.arange(W + 1)
#     yy = np.arange(H + 1)
#     im = ax.pcolormesh(xx, yy, C, shading="auto", vmin=vmin, vmax=vmax)
#     ax.set_xlabel("x"); ax.set_ylabel("y")
#     ax.set_title(title)
#     return im

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os


def plot_beta_histograms_by_channel(
    beta: np.ndarray,                # (T, C, Kmax)
    n_modes_per_channel,            # dict or list
    channel: int,
    top_n: int = 10,
    bin_width: float = 0.5,
    bound: float = 10.0,
    out_dir: str = "./plots/beta_histogram",
    show_others: bool = True,
):
    """
    Plot one histogram for one channel.

    Range is fixed to [-bound, bound].
    Values < -bound are clipped into the leftmost bin.
    Values > +bound are clipped into the rightmost bin.

    For each bin, choose the top_n modes by count within that bin.
    """

    T, C, Kmax = beta.shape
    c = channel
    k = int(n_modes_per_channel[c])

    b = beta[:, c, :k]   # (T, k)

    # edges for [-bound, bound] with interval bin_width
    edges = np.arange(-bound, bound + bin_width, bin_width)
    nbins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    # mode_counts[m, j] = count of mode m in bin j
    mode_counts = np.zeros((k, nbins), dtype=np.int64)

    for m in range(k):
        vals = b[:, m].copy()

        # clip tails into boundary bins
        vals = np.clip(vals, -bound, bound)

        hist_m, _ = np.histogram(vals, bins=edges)
        mode_counts[m] = hist_m

    cmap = plt.get_cmap("hsv", k)
    mode_colors = [cmap(i) for i in range(k)]
    others_color = (0.75, 0.75, 0.75, 1.0)

    fig, ax = plt.subplots(figsize=(16, 7))

    for j in range(nbins):
        counts_j = mode_counts[:, j]
        total_j = counts_j.sum()
        if total_j == 0:
            continue

        top_idx = np.argsort(counts_j)[::-1][:min(top_n, k)]
        top_idx = [m for m in top_idx if counts_j[m] > 0]

        bottom = 0.0

        for m in top_idx:
            h = counts_j[m]
            ax.bar(
                centers[j],
                h,
                width=bin_width,
                bottom=bottom,
                color=mode_colors[m],
                edgecolor="none",
                align="center",
            )
            bottom += h

        if show_others:
            others = total_j - sum(counts_j[m] for m in top_idx)
            if others > 0:
                ax.bar(
                    centers[j],
                    others,
                    width=bin_width,
                    bottom=bottom,
                    color=others_color,
                    edgecolor="none",
                    align="center",
                )

    # clipped stats only for display clarity
    all_vals = b.reshape(-1)
    clipped_vals = np.clip(all_vals, -bound, bound)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlim(-bound, bound)
    ax.set_title(
        f"Channel {c:02d} beta histogram "
        f"(top {top_n} modes per bin, clipped to [{-bound}, {bound}], "
        f"valid modes={k}, mean={all_vals.mean():.3f}, std={all_vals.std():.3f})"
    )
    ax.set_xlabel("beta")
    ax.set_ylabel("count")

    legend_handles = [
        Patch(facecolor=mode_colors[m], edgecolor="none", label=f"mode {m+1}")
        for m in range(k)
    ]
    if show_others:
        legend_handles.append(
            Patch(facecolor=others_color, edgecolor="none", label="others")
        )

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=7,
        ncol=2,
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"beta_histogram_channel_{channel}.png")
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        plt.show()
def plot_pca_validation(res_gt, res_pred, res_calib, title="calibrated", save_path="./plots", c=0, plot_idx = [0, 10, 20, 30, 40, 50]):
    channels = ["50u", "50v", "75u", "75v", "100u", "100v", "125u", "125v", "150u", "150v"]
    T = plot_idx # only plot the first 3 time steps
    res_pred_mean = np.mean(res_pred, axis=0)
    res_calib_mean = np.mean(res_calib, axis=0)
    # res_calib_mean=res_calib
    for t in plot_idx:
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        vmin, vmax = plot_minmax(res_gt[c, t],res_calib_mean[c, t] )
        plot_field(ax[0, 0], res_gt[c, t], "Original "+channels[c], vmin=vmin, vmax=vmax)
        # plot_field(ax[0, 1], res_pred_mean[c, t], "Predicted mean "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 2], res_calib_mean[c, t], "Calibrated mean"+channels[c], vmin=vmin, vmax=vmax)
        
        # for i in range(3):
        #     plot_field(ax[1, i], res_pred[i, c, t], f"Predicted ens {i}"+channels[c], vmin=vmin, vmax=vmax)
        #     plot_field(ax[2, i], res_calib[i, c, t], f"Calibrated ens {i}"+channels[c], vmin=vmin, vmax=vmax)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/pca_{title}_plot_{t}.png")
        plt.close()
    return fig, ax
def plot_predictions(inp, gt, reg, gt_res, res_pred, res_calib, title="calibrated_predictions", save_path="./plots", c=0, plot_idx = 3):
    channels = ["50u", "50v", "75u", "75v", "100u", "100v", "125u", "125v", "150u", "150v"]
    T = plot_idx # only plot the first 3 time steps
    res_pred_mean = np.mean(res_pred, axis=0)
    res_calib_mean = np.mean(res_calib, axis=0)
    for t in range(T):
        fig, ax = plt.subplots(3, 4, figsize=(20, 15))
        vmin, vmax = plot_minmax(inp[c, t], gt[c, t], reg[c, t], gt_res[c, t] )
        plot_field(ax[0, 0], inp[c, t], "Input LR"+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 1], gt[c, t], "GT HR "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 2], reg[c, t], "Regression "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 3], gt_res[c, t], "GT residual "+channels[c], vmin=vmin, vmax=vmax)
        
        for i in range(4):
            plot_field(ax[1, i], res_pred[i, c, t]+reg[c, t], f"Predicted ens {i}"+channels[c], vmin=vmin, vmax=vmax)
            plot_field(ax[2, i], res_calib[i, c, t]+reg[c, t], f"Calibrated ens {i}"+channels[c], vmin=vmin, vmax=vmax)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/pca_{title}_plot_{t}.png")
        plt.close()
    return fig, ax

def plot_corrdiff(gt, reg, gt_res, res_pred, title="corrdiff", save_path="./plots", c=0):
    channels = ["50u", "50v", "75u", "75v", "100u", "100v", "125u", "125v", "150u", "150v"]
    T = 3 # only plot the first 3 time steps
    res_pred_mean = np.mean(res_pred, axis=0)
    pred = res_pred + reg
    pred_mean = np.mean(pred, axis=0)
    for t in range(T):
        fig, ax = plt.subplots(3, 5, figsize=(20, 15))
        vmin, vmax = plot_minmax(res_pred_mean[c, t], gt[c, t], reg[c, t], gt_res[c, t] )
        plot_field(ax[0, 0], gt[c, t], "GT HR "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 1], pred_mean[c, t], "Prediction mean "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 2], reg[c, t], "Regression "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 3], gt_res[c, t], "GT residual "+channels[c], vmin=vmin, vmax=vmax)
        plot_field(ax[0, 4], res_pred_mean[c, t], "residual mean "+channels[c], vmin=vmin, vmax=vmax)
        
        for i in range(5):
            plot_field(ax[1, i], pred[i, c, t], "prediction ens " + str(i) + " " + channels[c], vmin=vmin, vmax=vmax)
            plot_field(ax[2, i], res_pred[i, c, t], "residual ens " + str(i) + " " + channels[c], vmin=vmin, vmax=vmax)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{title}_plot_{t}.png")
        plt.close()
    return fig, ax
# =========================================================
# Plot: all modes in one figure for one channel
# =========================================================


def plot_channel_all_mode_weights_scatter(
    train_weights: Dict[int, np.ndarray],
    val_gt_weights: Dict[int, np.ndarray],
    val_weights: Dict[int, np.ndarray],
    channel: int,
    max_points_per_mode_train: Optional[int] = 1000,
    max_points_per_mode_val: Optional[int] = 3000,
    max_points_per_mode_val_gt: Optional[int] = 1000,
    jitter: float = 0.12,
    alpha_train: float = 0.18,
    alpha_val: float = 0.12,
    s_train: float = 6.0,
    s_val: float = 5.0,
    s_val_gt: float = 6.0,
    alpha_val_gt: float = 0.18,
    random_state: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
    lagend: list[str] = ["train_gt", "val_gt", "val_pred"],
):
    """
    Plot all PCA modes for one channel on a single scatter plot.

    Expected shapes
    ---------------
    train_weights[channel]  : (T_train, K)
    val_gt_weights[channel] : (T_val, K)
    val_weights[channel]    : either
                              - (T_val, K), or
                              - (S, T_val, K)

    If val_weights[channel] is (S, T, K), all ensemble members and all times
    are scattered together for each mode.
    """
    if channel not in train_weights:
        raise KeyError(f"channel {channel} not found in train_weights")
    if channel not in val_weights:
        raise KeyError(f"channel {channel} not found in val_weights")
    if channel not in val_gt_weights:
        raise KeyError(f"channel {channel} not found in val_gt_weights")

    Wtr = np.asarray(train_weights[channel])     # (T_train, K)
    Wva = np.asarray(val_weights[channel])       # (T_val, K) or (S, T_val, K)
    Wva_gt = np.asarray(val_gt_weights[channel]) # (T_val, K)

    if Wtr.ndim != 2:
        raise ValueError(f"train_weights[channel] must have shape (T, K), got {Wtr.shape}")
    if Wva_gt.ndim != 2:
        raise ValueError(f"val_gt_weights[channel] must have shape (T, K), got {Wva_gt.shape}")
    if Wva.ndim not in (2, 3):
        raise ValueError(f"val_weights[channel] must have shape (T, K) or (S, T, K), got {Wva.shape}")

    # infer K consistently
    if Wva.ndim == 2:
        K_val = Wva.shape[1]
    else:
        K_val = Wva.shape[2]
    K = min(Wtr.shape[1], Wva_gt.shape[1], K_val)

    if Wtr.shape[1] != K or Wva_gt.shape[1] != K or K_val != K:
        print(
            f"[WARN] mode counts differ: "
            f"train={Wtr.shape[1]}, val_gt={Wva_gt.shape[1]}, val={K_val}. "
            f"Using first K={K} modes."
        )

    rng = np.random.default_rng(random_state)

    plt.figure(figsize=(max(10, K * 0.22), 6.5))

    train_labeled = False
    val_labeled = False
    val_gt_labeled = False

    for k in range(K):
        # train: (T_train,)
        y_tr = Wtr[:, k]

        # val_gt: (T_val,)
        y_va_gt = Wva_gt[:, k]

        # val_pred:
        # if (T, K) -> (T,)
        # if (S, T, K) -> flatten to (S*T,)
        if Wva.ndim == 2:
            y_va = Wva[:, k]
        else:
            y_va = Wva[:, :, k].reshape(-1)

        # subsample for readability
        if max_points_per_mode_train is not None and len(y_tr) > max_points_per_mode_train:
            idx_tr = rng.choice(len(y_tr), size=max_points_per_mode_train, replace=False)
            y_tr = y_tr[idx_tr]

        if max_points_per_mode_val is not None and len(y_va) > max_points_per_mode_val:
            idx_va = rng.choice(len(y_va), size=max_points_per_mode_val, replace=False)
            y_va = y_va[idx_va]

        if max_points_per_mode_val_gt is not None and len(y_va_gt) > max_points_per_mode_val_gt:
            idx_va_gt = rng.choice(len(y_va_gt), size=max_points_per_mode_val_gt, replace=False)
            y_va_gt = y_va_gt[idx_va_gt]

        # x jitter
        x_tr = k - 0.2 + rng.uniform(-jitter, jitter, size=len(y_tr))
        x_va = k       + rng.uniform(-jitter, jitter, size=len(y_va))
        x_va_gt = k + 0.2 + rng.uniform(-jitter, jitter, size=len(y_va_gt))

        plt.scatter(
            x_tr, y_tr,
            s=s_train,
            color="blue",
            alpha=alpha_train,
            label=lagend[0] if not train_labeled else None,
        )
        plt.scatter(
            x_va, y_va,
            s=s_val,
            color="green",
            alpha=alpha_val,
            label=lagend[2] if not val_labeled else None,
        )
        plt.scatter(
            x_va_gt, y_va_gt,
            s=s_val_gt,
            color="red",
            alpha=alpha_val_gt,
            label=lagend[1] if not val_gt_labeled else None,
        )

        train_labeled = True
        val_labeled = True
        val_gt_labeled = True

    plt.axhline(0.0, linewidth=1.0, alpha=0.5)
    plt.xlabel("PCA mode index")
    plt.ylabel("Projection weight")
    plt.title(f"Channel {channel}: all-mode PCA weight distribution")
    plt.xticks(np.arange(K))
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# def ds_to_array(ds: xr.Dataset, name: str = "variable") -> xr.DataArray:
#     if len(ds.data_vars) == 0:
#         raise ValueError("Dataset has no data variables.")
#     return ds.to_array(dim=name)


def load_nc_data_ot(nc_path: str, max_time_idx: int = None, return_all: bool = False):
    """
    Load validation data from NetCDF file.

    Target output shapes:
        gt_res_np : (C, T, H, W)
        res_np    : (S, C, T, H, W)

    Assumed dataset structure:
        truth/regression : per-var shape (time, y, x)
        prediction       : per-var shape (ensemble, time, y, x)
    """
    
    if max_time_idx is not None:
        t_dim = "time"
        idxs = np.arange(max_time_idx)

        gt_ds   = xr.open_dataset(nc_path, group="truth").isel({t_dim: idxs})
        reg_ot_ds = xr.open_dataset(nc_path, group="prediction").isel({t_dim: idxs})

        inp = xr.open_dataset(nc_path, group="input").isel({t_dim: idxs})
    else:
        gt_ds   = xr.open_dataset(nc_path, group="truth")
        reg_ot_ds = xr.open_dataset(nc_path, group="prediction")
        inp = xr.open_dataset(nc_path, group="input")
    # prediction has ensemble, regression does not.
    # xarray will broadcast regression over ensemble automatically.

    # truth/regression -> (C, T, H, W)
    gt_arr = ds_to_array(gt_ds)
    reg_ot_arr = ds_to_array(reg_ot_ds)
    # gt_res_ot_arr = gt_arr - reg_ot_arr

    gt_np = np.array(gt_arr)            # (C, T, H, W)
    reg_ot_np = np.array(reg_ot_arr)
    if len(reg_ot_np.shape) == 5:
        reg_ot_np = reg_ot_np.transpose(1, 0, 2, 3, 4)
        reg_ot_np=reg_ot_np[0]
    gt_res_ot_np = gt_np - reg_ot_np
    
    
    C, T, H, W = gt_np.shape
    print(f"gt np shape : {gt_np.shape}")
    print(f"reg ot np shape : {reg_ot_np.shape}")
    print(f"gt res ot np shape : {gt_res_ot_np.shape}")

    # print(f"max_time_idx: {max_time_idx}")
    print(f"Loaded {T} time steps from {nc_path}")
    # print(f"gt shape:      {gt_np.shape}")       # (C, T, H, W)
    # print(f"reg shape:     {reg_np.shape}")      # (C, T, H, W)
    # print(f"gt_res shape:  {gt_res_np.shape}")   # (C, T, H, W)
    # print(f"pred shape:    {pred_np.shape}")     # (S, C, T, H, W)
    # print(f"res shape:     {res_np.shape}")      # (S, C, T, H, W)

    if return_all:
        inp_np = np.array(ds_to_array(inp))
        return gt_res_ot_np, reg_ot_np, gt_np, inp_np
    return gt_res_ot_np

def plot_field(ax, C, title, vmin=None, vmax=None, isDiff=False):
    C = np.asarray(C)
    use_latlon=False
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
            im = ax.pcolormesh(lon2d, lat2d, C, shading="auto", vmin=vmin, vmax=vmax,transform=ccrs.PlateCarree(),cmap="RdBu_r")
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
def plot_ot_result_with_all_channels(ot_path: str, original_path: str, save_file: str):
    res_ot, reg_ot, gt_ot, inp_ot = load_nc_data_ot(ot_path, return_all=True, max_time_idx=10)
    res_original, reg_original, gt_original, inp_original = load_nc_data_ot(original_path, return_all=True, max_time_idx=10)
    C,T,H,W = gt_ot.shape
    channel_names=["50u", "50v","75u","75v","100u","100v","125u","125v","150u","150v"]
    
    for c in range(C):
        fig, axes = plt.subplots(2, 3, figsize=(20,10))
        plot_field(axes[0,0], inp_ot[c,0,:,:], vmin=np.min([(inp_ot[c,0,:,:].min(), inp_original[c,0,:,:].min())]), vmax=np.max([(inp_ot[c,0,:,:].max(), inp_original[c,0,:,:].max())]), title =f"input {channel_names[c]}")
        plot_field(axes[1,0], inp_original[c,0,:,:], vmin=np.min([(inp_ot[c,0,:,:].min(), inp_original[c,0,:,:].min())]), vmax=np.max([(inp_ot[c,0,:,:].max(), inp_original[c,0,:,:].max())]), title =f"input {channel_names[c]}")
        plot_field(axes[0,1], reg_ot[c,0,:,:], vmin=np.min([(reg_ot[c,0,:,:].min(), reg_original[c,0,:,:].min())]), vmax=np.max([(reg_ot[c,0,:,:].max(), reg_original[c,0,:,:].max())]), title=f"regression OT {channel_names[c]} ")
        plot_field(axes[1,1], reg_original[c,0,:,:], vmin=np.min([(reg_ot[c,0,:,:].min(), reg_original[c,0,:,:].min())]), vmax=np.max([(reg_ot[c,0,:,:].max(), reg_original[c,0,:,:].max())]), title =f"regression original {channel_names[c]}")
        plot_field(axes[0,2], res_ot[c,0,:,:], vmin=np.min([(res_ot[c,0,:,:].min(), res_original[c,0,:,:].min())]), vmax=np.max([(res_ot[c,0,:,:].max(), res_original[c,0,:,:].max())]), title=f"OT residual {channel_names[c]} ")
        plot_field(axes[1,2], res_original[c,0,:,:], vmin=np.min([(res_ot[c,0,:,:].min(), res_original[c,0,:,:].min())]), vmax=np.max([(res_ot[c,0,:,:].max(), res_original[c,0,:,:].max())]), title =f"original residual {channel_names[c]} ")
        os.makedirs(save_file, exist_ok=True)
        plt.savefig(f"{save_file}/{channel_names[c]}.png")
        plt.close()
def plot_compare(ot_path: str, original_path: str, save_file: str, channel_names, channel_idx=[0,2,4,6], idx =[123,666,8800,12000,17000]):
    res_ot, reg_ot, gt_ot, inp_ot = load_nc_data_ot(ot_path, return_all=True)
    res_original, reg_original, gt_original, inp_original = load_nc_data_ot(original_path, return_all=True)
    C,T,H,W = gt_ot.shape
    # print(f"shape res_ot: {res_ot.shape}")
    # print(f"shape res_original: {res_original.shape}")
    # print(f"shape reg_ot: {reg_ot.shape}")
    # print(f"shape reg_original: {reg_original.shape}")
    # print(f"shape gt_ot: {gt_ot.shape}")
    # print(f"shape gt_original: {gt_original.shape}")
    # print(f"shape inp_ot: {inp_ot.shape}")
    # print(f"shape inp_original: {inp_original.shape}")
    for i in (idx):
        fig, axes = plt.subplots(8, 4, figsize=(12,24))
        for c_i,c in enumerate(channel_idx):
            inp_now = inp_ot[c,i,:,:]
            gt_now = gt_ot[c,i,:,:]
            reg_ot_now = reg_ot[c,i,:,:]
            res_ot_now = res_ot[c,i,:,:]
            res_original_now = res_original[c,i,:,:]
            reg_original_now = reg_original[c,i,:,:]
            pred_original_now = res_ot_now + reg_original_now
            pred_ot_now = res_ot_now + reg_ot_now
            row0 = 2 * c_i
            row1 = 2 * c_i + 1
            min_val = np.min([inp_now.min(), gt_now.min(), reg_ot_now.min(), res_ot_now.min(), pred_ot_now.min(), res_original_now.min(), reg_original_now.min(), pred_original_now.min()])
            max_val = np.max([inp_now.max(), gt_now.max(), reg_ot_now.max(), res_ot_now.max(), pred_ot_now.max(), res_original_now.max(), reg_original_now.max(), pred_original_now.max()])
            plot_field(axes[row0, 0], inp_now, vmin=min_val, vmax=max_val, title =f"input {channel_names[c]}")
            plot_field(axes[row0, 1], reg_original_now, vmin=min_val, vmax=max_val, title =f"regression original {channel_names[c]}")
            plot_field(axes[row0, 2], res_original_now, vmin=min_val, vmax=max_val, title =f"original residual {channel_names[c]}")
            plot_field(axes[row0, 3], pred_original_now, vmin=min_val, vmax=max_val, title =f"original prediction {channel_names[c]}")
            plot_field(axes[row1, 0], gt_now, vmin=min_val, vmax=max_val, title =f"truth {channel_names[c]}")
            plot_field(axes[row1, 1], reg_ot_now, vmin=min_val, vmax=max_val, title =f"regression OT {channel_names[c]}")
            plot_field(axes[row1, 2], res_ot_now, vmin=min_val, vmax=max_val, title =f"residual OT {channel_names[c]}")
            plot_field(axes[row1, 3], pred_ot_now, vmin=min_val, vmax=max_val, title =f"prediction OT {channel_names[c]}")
        file_path = f"{save_file}/{i}.png"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        plt.close()
        print(f"Saved {file_path}")

def plot_ot_result(res_ot, res_original, save_file: str, channel_names, channel_idx=[0,2,4,6], idx =[123,666,8800,12000,17000]):
    # res_ot, reg_ot, gt_ot, inp_ot = load_nc_data_ot(ot_path, return_all=True, max_time_idx=10)
    # res_original, reg_original, gt_original, inp_original = load_nc_data_ot(original_path, return_all=True, max_time_idx=10)
    for i in idx:
        fig, axes = plt.subplots(2, 4, figsize=(10, 10))
        plot_field(axes[0,0], res_ot[channel_idx[0],i,:,:], vmin=np.min([(res_ot[channel_idx[0],i,:,:].min(), res_original[channel_idx[0],i,:,:].min())]), vmax=np.max([(res_ot[channel_idx[0],i,:,:].max(), res_original[channel_idx[0],i,:,:].max())]), title=f"OT residual {channel_names[channel_idx[0]]},0")
        plot_field(axes[0,1], res_ot[channel_idx[1],i,:,:], vmin=np.min([(res_ot[channel_idx[1],i,:,:].min(), res_original[channel_idx[1],i,:,:].min())]), vmax=np.max([(res_ot[channel_idx[1],i,:,:].max(), res_original[channel_idx[1],i,:,:].max())]), title=f"OT residual {channel_names[channel_idx[1]]},0")
        plot_field(axes[0,2], res_ot[channel_idx[2],i,:,:], vmin=np.min([(res_ot[channel_idx[2],i,:,:].min(), res_original[channel_idx[2],i,:,:].min())]), vmax=np.max([(res_ot[channel_idx[2],i,:,:].max(), res_original[channel_idx[2],i,:,:].max())]), title=f"OT residual {channel_names[channel_idx[2]]},0")
        plot_field(axes[0,3], res_ot[channel_idx[3],i,:,:], vmin=np.min([(res_ot[channel_idx[3],i,:,:].min(), res_original[channel_idx[3],i,:,:].min())]), vmax=np.max([(res_ot[channel_idx[3],i,:,:].max(), res_original[channel_idx[3],i,:,:].max())]), title=f"OT residual {channel_names[channel_idx[3]]},0")
        plot_field(axes[1,0], res_original[channel_idx[0],i,:,:], vmin=np.min([(res_original[channel_idx[0],i,:,:].min(), res_ot[channel_idx[0],i,:,:].min())]), vmax=np.max([(res_original[channel_idx[0],i,:,:].max(), res_ot[channel_idx[0],i,:,:].max())]), title=f"original residual {channel_names[channel_idx[0]]},0")
        plot_field(axes[1,1], res_original[channel_idx[1],i,:,:], vmin=np.min([(res_original[channel_idx[1],i,:,:].min(), res_ot[channel_idx[1],i,:,:].min())]), vmax=np.max([(res_original[channel_idx[1],i,:,:].max(), res_ot[channel_idx[1],i,:,:].max())]), title=f"original residual {channel_names[channel_idx[1]]},0")
        plot_field(axes[1,2], res_original[channel_idx[2],i,:,:], vmin=np.min([(res_original[channel_idx[2],i,:,:].min(), res_ot[channel_idx[2],i,:,:].min())]), vmax=np.max([(res_original[channel_idx[2],i,:,:].max(), res_ot[channel_idx[2],i,:,:].max())]), title=f"original residual {channel_names[channel_idx[2]]},0")
        plot_field(axes[1,3], res_original[channel_idx[3],i,:,:], vmin=np.min([(res_original[channel_idx[3],i,:,:].min(), res_ot[channel_idx[3],i,:,:].min())]), vmax=np.max([(res_original[channel_idx[3],i,:,:].max(), res_ot[channel_idx[3],i,:,:].max())]), title=f"original residual {channel_names[channel_idx[3]]},0")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(f"{save_file}/{i}.png")
        print(f"Saved {save_file}/{i}.png")
        plt.close()