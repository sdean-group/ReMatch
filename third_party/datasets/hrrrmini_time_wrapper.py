import numpy as np
import torch

class TimeFlagWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        wrapped_dataset,
        time_source_dataset=None,
        cutoff_time=np.datetime64("2020-01-01T00:00:00", "ns"),
    ):
        self.wrapped_dataset = wrapped_dataset
        self.time_source_dataset = (
            wrapped_dataset if time_source_dataset is None else time_source_dataset
        )
        self.cutoff_ns = cutoff_time.astype(np.int64)

        if not hasattr(self.time_source_dataset, "times"):
            raise AttributeError("time_source_dataset must have a 'times' attribute.")

        if len(self.time_source_dataset.times) != len(self.wrapped_dataset):
            raise ValueError(
                f"len(time_source_dataset.times)={len(self.time_source_dataset.times)} "
                f"but len(wrapped_dataset)={len(self.wrapped_dataset)}. "
                "This wrapper assumes sample idx and time idx are 1:1."
            )

    def __len__(self):
        return len(self.wrapped_dataset)

    def get_item_time_ns(self, idx):
        return np.int64(self.time_source_dataset.times[idx])

    def get_item_is_after_cutoff(self, idx):
        return bool(self.get_item_time_ns(idx) >= self.cutoff_ns)

    def __getitem__(self, idx):
        item = self.wrapped_dataset[idx]
        is_after_cutoff = self.get_item_is_after_cutoff(idx)

        if isinstance(item, tuple):
            return (*item, is_after_cutoff)
        return item, is_after_cutoff

    # ---- forward dataset metadata APIs ----
    def input_channels(self):
        return self.wrapped_dataset.input_channels()

    def output_channels(self):
        return self.wrapped_dataset.output_channels()

    def image_shape(self):
        return self.wrapped_dataset.image_shape()

    def longitude(self):
        return self.wrapped_dataset.longitude()

    def latitude(self):
        return self.wrapped_dataset.latitude()

    def time(self):
        if hasattr(self.wrapped_dataset, "time"):
            return self.wrapped_dataset.time()
        return self.time_source_dataset.time()

    # optional: delegate any other unknown attributes automatically
    def __getattr__(self, name):
        return getattr(self.wrapped_dataset, name)