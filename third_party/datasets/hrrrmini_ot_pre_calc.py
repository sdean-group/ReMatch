import datetime
from typing import List, Tuple, Union
import json
import numpy as np
import xarray as xr

from physicsnemo.models.diffusion.training_utils import convert_datetime_to_cftime

from third_party.datasets.base import ChannelMetadata, DownscalingDataset

# self.data_path = "/data/experiment_outputs/calibration_purpose/tiny_optimal_transport/reg_2018-2020_ot.nc"
       
class HRRROTDataset(DownscalingDataset):
    """
    Reader for OT-saved dataset with groups:
      - truth
      - prediction
      - input
      - invariant   # 추가

    Assumptions
    -----------
    - truth/prediction: (C, T, H, W)
    - input: (C_in, T, H, W), already upsampled/full-resolution
    - invariant: (C_inv, H, W) or (C_inv, 1, H, W) depending on file
    """

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        invariant_variables: Union[List[str], None] = ("elev_mean", "lsm_mean"),
        return_truth_prediction: bool = False,
    ):
        self.data_path=data_path
        self.return_truth_prediction = return_truth_prediction

        # grouped datasets
        self.input, self.input_variables = _load_dataset(
            self.data_path, "input", input_variables
        )
        
        self.truth, self.output_variables = _load_dataset(
            self.data_path, "truth", output_variables
        )
        self.regression, self.regression_variables = _load_dataset(
            self.data_path, "prediction", output_variables
        )

        (self.invariants, self.invariant_variables) = _load_dataset(
            self.data_path, "input", invariant_variables
        )
        self.invariants = self.invariants[0]

        if self.output_variables != self.regression_variables:
            raise ValueError(
                f"truth variables {self.output_variables} and prediction variables {self.regression_variables} do not match"
            )

        if self.truth.shape != self.regression.shape:
            raise ValueError(
                f"truth shape {self.truth.shape} and prediction shape {self.regression.shape} must match"
            )

        self.img_shape = self.truth.shape[-2:]

        # time / coord
        with xr.open_dataset(self.data_path) as ds:
            self.times = np.array(ds["time"])

        # normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)

        input_mean, input_std = _load_stats(stats, self.input_variables, "input")
        inv_mean, inv_std = _load_stats(stats, self.invariant_variables, "invariant")
        self.input_mean = np.concatenate([input_mean, inv_mean], axis=0)
        self.input_std = np.concatenate([input_std, inv_std], axis=0)

        self.output_mean, self.output_std = _load_stats(
            stats, self.output_variables, "output"
        )

    def __getitem__(self, idx):
        """
        Return:
            truth_norm, x_norm, regression_norm
        where x includes invariant channels appended at the end.
        """
        x = self.input[idx].copy()           # (C_in, H, W)
        truth = self.truth[idx].copy()       # (C_out, H, W)
        reg = self.regression[idx].copy()   # (C_out, H, W)

        i=0
        j=0
        inv = self.invariants[:, i : i + self.img_shape[0], j : j + self.img_shape[1]]

        x = np.concatenate([x, inv], axis=0)

        x = self.normalize_input(x)
        truth = self.normalize_output(truth)
        reg = self.normalize_output(reg)

        return (truth, x, reg)

    def __len__(self):
        return self.input.shape[0]

    def longitude(self) -> np.ndarray:
        return np.full(self.img_shape, np.nan)

    def latitude(self) -> np.ndarray:
        return np.full(self.img_shape, np.nan)

    def input_channels(self) -> List[ChannelMetadata]:
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        invariants = [
            ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables
        ]
        return inputs + invariants

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        if isinstance(self.times[0], np.datetime64):
            datetimes = (
                datetime.datetime.utcfromtimestamp(
                    t.astype("datetime64[ns]").astype(np.int64) / 1e9
                )
                for t in self.times
            )
            return [convert_datetime_to_cftime(t) for t in datetimes]

        if np.issubdtype(np.array(self.times).dtype, np.number):
            return list(self.times)

        return list(self.times)

    def image_shape(self) -> Tuple[int, int]:
        return self.img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean
        
    def denormalize_residuals(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean


def _load_dataset(data_path, group, variables=None, stack_axis=1):
    with xr.open_dataset(data_path, group=group) as ds:
        if variables is None:
            variables = list(ds.keys())
        data = np.stack([ds[v] for v in variables], axis=stack_axis)
    return (data, variables)


def _load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(
        np.float32
    )
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(
        np.float32
    )
    return mean, std

