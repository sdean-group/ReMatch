from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from third_party.datasets.base import DownscalingDataset, ChannelMetadata



CHANNEL_NAMES = ["RHO", "UX", "UY", "UZ"]


class BLASTNetNCDataset(DownscalingDataset):
    """
    CorrDiff-compatible BLASTNet dataset.

    Expected NetCDF structure:
        root
        ├── input/
        │   └── data   : (N, C, H, W)
        ├── output/
        │   └── data   : (N, C, H, W)
        └── metadata/  : optional

    __getitem__ returns:
        img_clean : HR target, shape (C, H, W)
        img_lr    : upsampled LR input, shape (C, H, W)

    This follows CorrDiff's DownscalingDataset interface.
    """

    def __init__(
        self,
        data_path: str,
        input_group: str = "input",
        output_group: str = "output",
        variable_name: str = "data",
        multiple_input: bool = False,
        input_variable_name: str = "data", # when multiple_input is True, this is either input_level0, input_level1, input_level2, input_level3
        channel_names: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        print(f"******************input_variable_name: {input_variable_name}")
        self.data_path = Path(data_path)
        self.input_group = input_group
        self.output_group = output_group
        self.variable_name = variable_name
        self.input_variable_name = input_variable_name
        self.dtype = dtype

        self._nc = None
        self._input_var = None
        self._output_var = None

        self.multiple_input = multiple_input

        if channel_names is None:
            self.channel_names = CHANNEL_NAMES
        else:
            self.channel_names = channel_names

        if not self.data_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {self.data_path}")
        

        with NetCDFDataset(self.data_path, "r") as nc:
            if input_group not in nc.groups:
                raise KeyError(f"Group '{input_group}' not found in {self.data_path}")

            if output_group not in nc.groups:
                raise KeyError(f"Group '{output_group}' not found in {self.data_path}")

            if input_variable_name not in nc.groups[input_group].variables:
                raise KeyError(f"Variable '{input_group}/{input_variable_name}' not found")

            if variable_name not in nc.groups[output_group].variables:
                raise KeyError(f"Variable '{output_group}/{variable_name}' not found")

            input_shape = nc.groups[input_group].variables[input_variable_name].shape
            output_shape = nc.groups[output_group].variables[variable_name].shape

            if input_shape != output_shape:
                raise ValueError(
                    f"Input/output shape mismatch: input={input_shape}, output={output_shape}"
                )

            if len(input_shape) != 4:
                raise ValueError(
                    f"Expected shape (N, C, H, W), but got {input_shape}"
                )

            self.length = input_shape[0]
            self.num_channels = input_shape[1]
            self.height = input_shape[2]
            self.width = input_shape[3]

        if len(self.channel_names) != self.num_channels:
            raise ValueError(
                f"channel_names length mismatch: "
                f"got {len(self.channel_names)}, expected {self.num_channels}"
            )

        self._longitude = np.full((self.height, self.width), np.nan)
        self._latitude = np.full((self.height, self.width), np.nan)
        self._time = list(range(self.length))

        print(f"Loaded BLASTNet CorrDiff dataset: {self.data_path}")
        print(f"shape: ({self.length}, {self.num_channels}, {self.height}, {self.width})")
        # print(f"input group: {self.input_group}/{self.variable_name}")
        # print(f"output group: {self.output_group}/{self.variable_name}")

    def _lazy_open(self):
        """
        Open NetCDF lazily inside each worker.
        This is safer for DataLoader(num_workers > 0).
        """
        if self._nc is None:
            self._nc = NetCDFDataset(self.data_path, "r")
            self._input_var = self._nc.groups[self.input_group].variables[self.input_variable_name]
            self._output_var = self._nc.groups[self.output_group].variables[self.variable_name]

    def longitude(self) -> np.ndarray:
        return self._longitude

    def latitude(self) -> np.ndarray:
        return self._latitude

    def input_channels(self) -> List[ChannelMetadata]:
        return [
            ChannelMetadata(name=name, level="", auxiliary=False)
            for name in self.channel_names
        ]

    def output_channels(self) -> List[ChannelMetadata]:
        return [
            ChannelMetadata(name=name, level="", auxiliary=False)
            for name in self.channel_names
        ]

    def time(self) -> List:
        return self._time

    def image_shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def __len__(self) -> int:
        return self.length

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        # Already normalized to [0, 1] when the .nc file was generated.
        return x

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        # Cannot recover physical units unless min/max metadata is saved.
        return x

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        # Already normalized to [0, 1] when the .nc file was generated.
        return x

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        # Cannot recover physical units unless min/max metadata is saved.
        return x
    def denormalize_residuals(self, x: np.ndarray) -> np.ndarray:
        # Cannot recover physical units unless min/max metadata is saved.
        return x

    def __getitem__(self, idx: int):
        self._lazy_open()
        img_lr = self._input_var[idx]      # (C, H, W), upsampled LR
        img_clean = self._output_var[idx]  # (C, H, W), HR target

        img_lr = torch.as_tensor(np.asarray(img_lr), dtype=self.dtype)
        img_clean = torch.as_tensor(np.asarray(img_clean), dtype=self.dtype)

        return img_clean, img_lr

    def info(self) -> dict:
        return {
            "name": "BLASTNetNCDataset",
            "data_path": str(self.data_path),
            "length": self.length,
            "input_channels": self.channel_names,
            "output_channels": self.channel_names,
            "image_shape": (self.height, self.width),
            "normalization": "paired LR-HR min-max, already applied in NetCDF",
        }

    def close(self):
        if self._nc is not None:
            self._nc.close()
            self._nc = None
            self._input_var = None
            self._output_var = None

    def __del__(self):
        self.close()

        
def plot_lr_hr_matched_slices(y,x, idx=None, save_path=None):
    """
    Compare LR_8x z-slices with corresponding HR z-slices.
    LR z index: 0..15
    HR z index: z_lr * 8
    """
    C, H, W = x.shape

    fig, axes = plt.subplots(C, 2, figsize=(6, C*2), constrained_layout=True)

    for c in range(C):
        lr_slice = (x[c]).float()  # (16, 16)
        hr_slice = (y[c]).float()  # (128, 128)

        axes[c, 0].imshow(lr_slice, origin="lower", vmin=0, vmax=1)
        axes[c, 0].set_title(f"LR {CHANNEL_NAMES[c]}", fontsize=8)
        axes[c, 0].axis("off")

        axes[c, 1].imshow(hr_slice, origin="lower", vmin=0, vmax=1)
        axes[c, 1].set_title(f"HR {CHANNEL_NAMES[c]}", fontsize=8)
        axes[c, 1].axis("off")



        fig.suptitle(
            f"LR-HR matched slices | channel={CHANNEL_NAMES[c]}\n"
            f"idx={idx}",
            fontsize=10,
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()

if __name__ == "__main__":
    nc_path = "/data/blastnet_momentum/blastnet_train_lr16x_z8_paired_minmax.nc"
    dataset = BLASTNetNCDataset(nc_path)
    idx = [0,100,500,1000,4000,8000,10000]
    for i in idx:
        y, x = dataset.__getitem__(i)
        plot_lr_hr_matched_slices(y, x, idx=i, save_path=f"../plots/blastnet/normalized_train/lr_hr_matched_slices_{i}.png")
