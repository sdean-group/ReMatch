import numpy as np
from pathlib import Path
import xskillscore
import xarray as xr
import os


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


def save_rmse(out_path: Path, **kwargs):
    (out_path / "rmse").mkdir(parents=True, exist_ok=True)
    (out_path / "ssr").mkdir(parents=True, exist_ok=True)
    for i, (title, data) in enumerate(kwargs.items()):
        print(f"----- {title} --------")
        file_path = data[0]
        truth, pred, root, inp = open_samples(file_path)
        pred = pred.load()
        dim = ["x", "y"]
        b = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
        a = (pred.var("ensemble").mean(dim=["x", "y"])) ** 0.5
        spread_mean = a.mean(dim="time")
        rmse_mean = b.mean(dim="time")
        ssr = a/b
        ssr_mean = spread_mean / rmse_mean
        rmse_scalar = b.to_array().mean().item()
        ssr_scalar = a.to_array().mean().item() / b.to_array().mean().item()
        print(f"rmse: {rmse_scalar:.4f}, ssr: {ssr_scalar:.4f}")
        
        
        if os.path.exists(out_path / "rmse" / f"{title}_channelwise.txt"):
            print(f"Found channelwise rmse - skipping save")
        else:
            with open(out_path / "rmse" / f"{title}_channelwise.txt", "a") as f:
                for var in rmse_mean.data_vars:
                    value = rmse_mean[var].item()  # extract scalar
                    f.write(f"{var}: {value}\n")

        if os.path.exists(out_path / "rmse" / f"{title}.txt"):
            print(f"Found rmse - skipping save")
        else:
            b_scalar = b.to_array().mean().item()
            with open(out_path / "rmse" / f"{title}.txt", "a") as f:
                f.write(f"{b_scalar}\n")

        if os.path.exists(out_path / "rmse" / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path / "rmse" / f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            b.to_netcdf(OUT_NC, mode="w", engine="netcdf4")



        if os.path.exists(out_path / "ssr" / f"{title}_channelwise.txt"):
            print(f"Found channelwise ssr - skipping save")
        else:
            with open(out_path / "ssr" / f"{title}_channelwise.txt", "a") as f:
                for var in ssr_mean.data_vars:
                    value = ssr_mean[var].item()  # extract scalar
                    f.write(f"{var}: {value}\n")

        if os.path.exists(out_path / "ssr" / f"{title}.txt"):
            print(f"Found ssr - skipping save")
        else:
            ssr_scalar = a.to_array().mean().item() / b.to_array().mean().item()
            with open(out_path / "ssr" / f"{title}.txt", "a") as f:
                f.write(f"{ssr_scalar}\n")

        if os.path.exists(out_path / "ssr" / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path / "ssr" / f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            ssr.to_netcdf(OUT_NC, mode="w", engine="netcdf4")


out = Path("/mnt/hrrr_era5_0528/experiment_result/metrics_paper/rmseandssr")
os.makedirs(out, exist_ok=True)
base_dir = Path("/mnt/hrrr_era5_0528/experiment_result")
print(f"base dir : {base_dir}")
save_rmse(
    out_path=out,
    # unet_original=[str(base_dir / "unet" / "600_samples.nc")],
    # cdm=[str(base_dir / "cdm" / "600_samples.nc")],
    # corrdiff_m=[str(base_dir / "corrdiff_m" / "600_samples.nc")],
    # corrdiff=[str(base_dir / "corrdiff" / "600_samples.nc")],
    # convfno=[str(base_dir / "convfno" / "600_samples.nc")],
    # swinir=[str(base_dir / "swinir" / "600_samples.nc")],
    # cfg=[str(base_dir / "cfg" / "600_samples.nc")],
    # # uq_rmse_0625=[str(base_dir / "uq_rmse_0625" / "600_samples.nc")],
    # # uq_quantiles_original=[str(base_dir / "uq_quantiles_nidhi" / "600_samples.nc")],
    # uq_quantiles_gt=[str(base_dir / "uq_quantiles_gt" / "600_samples.nc")],
    # # uq_quantiles_zero_masking=[str(base_dir / "uq_quantiles_zero" / "600_samples.nc")],
    # uq_rmse_original=[str(base_dir / "uq_rmse_0625" / "600_samples.nc")],
    # # uq_rmse_zero=[str(base_dir / "uq_rmse_zero" / "600_samples.nc")],
    # rematch_u=[str(base_dir / "rematch_u" / "600_samples.nc")],
    rematch_s=[str(base_dir / "rematch_s" / "rematchs_again_600_samples.nc")],
)