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


def save_crps(out_path: Path, **kwargs):
    for i, (title, data) in enumerate(kwargs.items()):
        print(f"----- {title} --------")
        if (os.path.exists(out_path / "crps" / f"{title}_channelwise.txt") and os.path.exists(out_path / "crps" / f"{title}.txt") and os.path.exists(out_path / "crps" f"{title}.nc")):
            print(f"All stats exist, moving to next one")
            continue
        file_path = data[0]
        truth, pred, root, inp = open_samples(file_path)
        pred = pred.load()
        dim = ["x", "y"]
        
        b = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)
        print(b.to_array().mean().item())
        
        if os.path.exists(out_path / f"{title}_channelwise.txt"):
            print(f"Found channelwise metric - skipping save")
        else:
            b_mean = b.mean(dim="time")
            with open(out / f"{title}_channelwise.txt", "a") as f:
                for var in b_mean.data_vars:
                    value = b_mean[var].item()  # extract scalar
                    f.write(f"{var}: {value}\n")

        if os.path.exists(out_path / f"{title}.txt"):
            print(f"Found metric - skipping save")
        else:
            b_scalar = b.to_array().mean().item()
            with open(out / f"{title}.txt", "a") as f:
                f.write(f"{b_scalar}\n")

        if os.path.exists(out_path / f"{title}.nc"):
            print(f"Found metrics NC file - skipping save")
        else:
            OUT_NC = out_path / f"{title}.nc"
            OUT_NC.parent.mkdir(parents=True, exist_ok=True)
            if OUT_NC.exists():
                OUT_NC.unlink()
            
            b.to_netcdf(OUT_NC, mode="w", engine="netcdf4")
        print("---------------")

# verified that crps for deterministic models equals the MAE

out = Path("/mnt/hrrr_era5_0528/experiment_result/metrics_paper/crps")
os.makedirs(out, exist_ok=True)
base_dir = Path("/mnt/hrrr_era5_0528/experiment_result")

save_crps(
    out_path=out,
    # unet=[str(base_dir / "unet" / "600_samples.nc")],
    # cdm=[str(base_dir / "cdm" / "600_samples.nc")],
    # corrdiff_m=[str(base_dir / "corrdiff_m" / "600_samples.nc")],
    # corrdiff=[str(base_dir / "corrdiff" / "600_samples.nc")],
    # convfno=[str(base_dir / "convfno" / "600_samples.nc")],
    # swinir=[str(base_dir / "swinir" / "600_samples.nc")],
    # cfg=[str(base_dir / "cfg" / "600_samples.nc")],
    # uq_rmse_gt=[str(base_dir / "uq_rmse_0625" / "600_samples.nc")],
    # uq_quantiles_gt=[str(base_dir / "uq_quantiles_gt" / "600_samples.nc")],
    # rematch_u=[str(base_dir / "rematch_u" / "600_samples.nc")],
    # rematch_s=[str(base_dir / "rematch_s" / "600_samples.nc")],
    rematch_s_correct=[str(base_dir / "rematch_s" / "rematchs_again_600_samples.nc")],
)