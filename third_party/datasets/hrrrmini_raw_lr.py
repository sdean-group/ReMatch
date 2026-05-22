from tkinter.constants import X
from third_party.datasets.hrrrmini import HRRRMiniDataset
from typing import Union, List
import numpy as np
import torch
class HRRRMiniRawLRDataset(HRRRMiniDataset):
    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables=None,
        output_variables=None,
    ):

        super().__init__(
            data_path=data_path,
            stats_path=stats_path,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        
        self.input_raw_mean = self.input_mean[:len(input_variables)]
        self.input_raw_std = self.input_std[:len(input_variables)]
        
    def __getitem__(self, idx):
        """Return the data sample (output, input) at index idx."""
        raw = self.input[idx].copy()        
        x = self.upsample(raw)
        zeros = np.zeros(
            (2, x.shape[-2], x.shape[-1]),
            dtype=x.dtype,
        )

        y = self.output[idx]

        x = np.concatenate([x, zeros], axis=0)
        x = self.normalize_input(x)
        raw=self.normalize_raw_input(raw)
        y = self.normalize_output(y)
        # print(f"input x.shape: {x.shape}, output y.shape: {y.shape}")
        return (y, raw, x)
    def normalize_raw_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_raw_mean) / self.input_raw_std

    def denormalize_raw_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_raw_std + self.input_raw_mean
       