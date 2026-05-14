# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from physicsnemo.models.diffusion.training_utils import convert_datetime_to_cftime

from third_party.datasets.dataset import init_dataset_from_config
from third_party.datasets.base import DownscalingDataset
import cftime


def get_dataset_and_sampler(dataset_cfg, times, has_lead_time=False, load_all_times=False):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    
    all_times = dataset.time()
    if load_all_times:
        plot_times = all_times
    else:
        plot_times = [
            convert_datetime_to_cftime(
                datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
            )
            for time in times
        ]
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler
def get_dataset_and_sampler_timeindices(dataset_cfg, time_indices=None, load_all_times=True):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    
    all_times = dataset.time()
    
    if load_all_times:
        sampler = all_times
    else:
        sampler = time_indices

    return dataset, sampler


class CustomNetCDFWriter:
    """NetCDF Writer"""

    def __init__(
        self, f, lat, lon, input_channels, output_channels, has_lead_time=False, save_time=True, save_data_index=True
    ):
        self._f = f
        self.has_lead_time = has_lead_time
        # create unlimited dimensions
        f.createDimension("time")
        f.createDimension("ensemble")

        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have the same shape")
        ny, nx = lat.shape

        # create lat/lon grid
        f.createDimension("x", nx)
        f.createDimension("y", ny)

        v = f.createVariable("lat", "f", dimensions=("y", "x"))
        # NOTE rethink this for datasets whose samples don't have constant lat-lon.
        v[:] = lat
        v.standard_name = "latitude"
        v.units = "degrees_north"

        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon
        v.standard_name = "longitude"
        v.units = "degrees_east"

        # create time dimension
        if save_time:
            if has_lead_time:
                v = f.createVariable("time", "str", ("time"))
            else:
                v = f.createVariable("time", "i8", ("time"))
                v.calendar = "standard"
                v.units = "hours since 1990-01-01 00:00:00"

        if save_data_index:
            v = f.createVariable("data_index", "i8", ("time"))
            v.long_name = "original_dataset_index"
            v.description = "Original dataset index used to load each generated sample"
      
        self.truth_group = f.createGroup("truth")
        self.prediction_group = f.createGroup("prediction")
        self.input_group = f.createGroup("input")
        self.residual_group = f.createGroup("residual")
        self.regression_group = f.createGroup("regression")

        for variable in output_channels:
            name = variable.name + variable.level
            self.truth_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.prediction_group.createVariable(
                name, "f", dimensions=("ensemble", "time", "y", "x")
            )
            self.residual_group.createVariable(name, "f", dimensions=("ensemble", "time", "y", "x"))
            self.regression_group.createVariable(name, "f", dimensions=("time", "y", "x"))

        # setup input data in netCDF

        for variable in input_channels:
            name = variable.name + variable.level
            self.input_group.createVariable(name, "f", dimensions=("time", "y", "x"))

    def write_input(self, channel_name, time_index, val):
        """Write input data to NetCDF file."""
        self.input_group[channel_name][time_index] = val

    def write_truth(self, channel_name, time_index, val):
        """Write ground truth data to NetCDF file."""
        self.truth_group[channel_name][time_index] = val

    def write_prediction(self, channel_name, time_index, ensemble_index, val):
        """Write prediction data to NetCDF file."""
        self.prediction_group[channel_name][ensemble_index, time_index] = val
    def write_regression(self, channel_name, time_index, val):
        """Write ground truth data to NetCDF file."""
        self.regression_group[channel_name][time_index] = val

    def write_residual(self, channel_name, time_index, ensemble_index, val):
        """Write prediction data to NetCDF file."""
        self.residual_group[channel_name][ensemble_index, time_index] = val

    def write_time(self, time_index, time):
        """Write time information to NetCDF file."""
        # print(f"writing time: {time} at time index: {time_index}")
        if self.has_lead_time:
            self._f["time"][time_index] = time
        else:
            time_v = self._f["time"]
            self._f["time"][time_index] = cftime.date2num(
                time, time_v.units, time_v.calendar
            )
    def write_data_index(self, data_index, time_index):
        """Write data index information to NetCDF file."""
        self._f["data_index"][time_index] = data_index


def custom_save_images(
    writer: CustomNetCDFWriter,
    dataset: DownscalingDataset,
    times,
    image_out,
    image_tar,
    image_lr,
    time_index,
    dataset_index,
    image_res=None,
    image_reg=None,
    save_time=True,
    save_data_index=True,
):
    """
    Saves inferencing result along with the baseline

    Parameters
    ----------

    writer (NetCDFWriter): Where the data is being written
    in_channels (List): List of the input channels being used
    input_channel_info (Dict): Description of the input channels
    out_channels (List): List of the output channels being used
    output_channel_info (Dict): Description of the output channels
    input_norm (Tuple): Normalization data for input
    target_norm (Tuple): Normalization data for the target
    image_out (torch.Tensor): Generated output data
    image_tar (torch.Tensor): Ground truth data
    image_lr (torch.Tensor): Low resolution input data
    time_index (int): Epoch number
    dataset_index (int): index where times are located
    """
    # weather sub-plot
    image_lr2 = image_lr[0].unsqueeze(0)
    image_lr2 = image_lr2.cpu().numpy()
    image_lr2 = dataset.denormalize_input(image_lr2)

    image_tar2 = image_tar[0].unsqueeze(0)
    image_tar2 = image_tar2.cpu().numpy()
    image_tar2 = dataset.denormalize_output(image_tar2)

    # some runtime assertions
    
    if image_tar2.ndim != 4:
        raise ValueError("image_tar2 must be 4-dimensional")
    # print(f"image_reg shape: {image_reg.shape}")
    # print(f"image_out shape: {image_out.shape}")
    # print(f"image_res shape: {image_res.shape}")
    for idx in range(image_out.shape[0]):
        image_out2 = image_out[idx].unsqueeze(0)
        if image_out2.ndim != 4:
            raise ValueError("image_out2 must be 4-dimensional")
        # Denormalize the input and outputs
        image_out2 = image_out2.cpu().numpy()
        
        image_out2 = dataset.denormalize_output(image_out2)
        if image_res is not None:
            image_res2 = image_res[idx].unsqueeze(0)
            if image_res2.ndim != 4:
                raise ValueError("image_res2 must be 4-dimensional")
            image_res2 = image_res2.cpu().numpy()
            image_res2 = dataset.denormalize_residuals(image_res2)
        if image_reg is not None:
            image_reg2 = image_reg[idx].unsqueeze(0)
            if image_reg2.ndim != 4:
                raise ValueError("image_reg2 must be 4-dimensional")
            image_reg2 = image_reg2.cpu().numpy()
            image_reg2 = dataset.denormalize_output(image_reg2)
        if save_data_index:
            writer.write_data_index(dataset_index, time_index)
        if save_time:
            time = times[dataset_index]
            writer.write_time(time_index, time)
        for channel_idx in range(image_out2.shape[1]):
            info = dataset.output_channels()[channel_idx]
            channel_name = info.name + info.level
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction(
                channel_name, time_index, idx, image_out2[0, channel_idx]
            )
            if image_res is not None:
                writer.write_residual(
                    channel_name, time_index, idx, image_res2[0, channel_idx]
                )
            if image_reg is not None:
                writer.write_regression(
                    channel_name, time_index, image_reg2[0, channel_idx]
                )

    input_channel_info = dataset.input_channels()
    for channel_idx in range(len(input_channel_info)):
        info = input_channel_info[channel_idx]
        channel_name = info.name + info.level
        writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
        if channel_idx == image_lr2.shape[1] - 1:
            break

def save_images(
    writer,
    dataset: DownscalingDataset,
    times,
    image_out,
    image_tar,
    image_lr,
    time_index,
    dataset_index,
):
    """
    Saves inferencing result along with the baseline

    Parameters
    ----------

    writer (NetCDFWriter): Where the data is being written
    in_channels (List): List of the input channels being used
    input_channel_info (Dict): Description of the input channels
    out_channels (List): List of the output channels being used
    output_channel_info (Dict): Description of the output channels
    input_norm (Tuple): Normalization data for input
    target_norm (Tuple): Normalization data for the target
    image_out (torch.Tensor): Generated output data
    image_tar (torch.Tensor): Ground truth data
    image_lr (torch.Tensor): Low resolution input data
    time_index (int): Epoch number
    dataset_index (int): index where times are located
    """
    # weather sub-plot
    image_lr2 = image_lr[0].unsqueeze(0)
    image_lr2 = image_lr2.cpu().numpy()
    image_lr2 = dataset.denormalize_input(image_lr2)

    image_tar2 = image_tar[0].unsqueeze(0)
    image_tar2 = image_tar2.cpu().numpy()
    image_tar2 = dataset.denormalize_output(image_tar2)

    # some runtime assertions
    if image_tar2.ndim != 4:
        raise ValueError("image_tar2 must be 4-dimensional")

    for idx in range(image_out.shape[0]):
        image_out2 = image_out[idx].unsqueeze(0)
        if image_out2.ndim != 4:
            raise ValueError("image_out2 must be 4-dimensional")

        # Denormalize the input and outputs
        image_out2 = image_out2.cpu().numpy()
        image_out2 = dataset.denormalize_output(image_out2)
        # print(f"times: {times}, \n\n dataset_index: {dataset_index}")
        time = times[dataset_index]
        writer.write_time(time_index, time)
        for channel_idx in range(image_out2.shape[1]):
            info = dataset.output_channels()[channel_idx]
            channel_name = info.name + info.level
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction(
                channel_name, time_index, idx, image_out2[0, channel_idx]
            )

    input_channel_info = dataset.input_channels()
    for channel_idx in range(len(input_channel_info)):
        info = input_channel_info[channel_idx]
        channel_name = info.name + info.level
        writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
        if channel_idx == image_lr2.shape[1] - 1:
            break
