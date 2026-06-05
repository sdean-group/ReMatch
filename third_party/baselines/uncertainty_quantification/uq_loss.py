from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from physicsnemo.utils.patching import RandomPatching2D

from physicsnemo.metrics.diffusion import ResidualLoss
import math
from numba import jit, prange
import numpy as np


class RegressionLossBiasCorrector_rmse_q90:
    """
    Regression loss function for the bias corrector.
    Predict channel-wise RMSE and Q90 of residual bias.
    Note: this loss does not apply any reduction.
    """

    def __init__(self, regression_net):
        self.regression_net = regression_net
        self.regression_net.eval()
        return

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        w_rmse: float = 10.0,
        w_q90: float = 1.0,
    ) -> torch.Tensor:

        B, C, H, W = img_clean.shape

        img_tot = torch.cat((torch.zeros_like(img_clean), img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, :C, :, :]

        with torch.no_grad():
            img_mean = self.regression_net(
                torch.zeros_like(y, device=img_clean.device),
                img_lr,
                augment_labels=augment_labels,
            )

        gt_bias = img_clean - img_mean  # (B, C, H, W)

        # Channel-wise RMSE
        gt_bias_rmse = torch.sqrt((gt_bias ** 2).mean(dim=(-2, -1), keepdim=False))  # (B, C)

        # Channel-wise Q90 of |residual|
        gt_bias_abs = gt_bias.abs().flatten(start_dim=2)  # (B, C, H*W)
        gt_bias_q90 = torch.quantile(gt_bias_abs, q=0.9, dim=2)  # (B, C)

        img_tot = torch.cat((gt_bias, img_lr, img_mean), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, :img_clean.shape[1], :, :]
        y_conditioned = y_tot[:, img_clean.shape[1]:, :, :]

        zero_input = torch.zeros((B, 2 * C, H, W), device=img_clean.device)

        pred_bias = net(
            zero_input,
            y_conditioned,
            force_fp32=False,
            augment_labels=augment_labels,
        )

        pred_vec = F.adaptive_avg_pool2d(pred_bias, 1)      # (B, 2C, 1, 1)
        pred_vec = pred_vec.squeeze(-1).squeeze(-1)         # (B, 2C)
        pred_vec = pred_vec.view(pred_vec.shape[0], C, 2)   # (B, C, 2)

        pred_rmse = F.softplus(pred_vec[..., 0])            # (B, C)
        pred_q90 = F.softplus(pred_vec[..., 1])             # (B, C)

        err_rmse = (torch.log(pred_rmse + 1e-6) - torch.log(gt_bias_rmse + 1e-6)) ** 2
        err_q90 = (torch.log(pred_q90 + 1e-6) - torch.log(gt_bias_q90 + 1e-6)) ** 2

        loss_rmse = err_rmse.mean(dim=1)  # (B,)
        loss_q90 = err_q90.mean(dim=1)    # (B,)
        # train set: loss_rmse: 5.634230613708496, loss_q90: 3.5643091201782227, gt_rmse: 0.0666382908821106, gt_q90: 0.10868420451879501
        # calibration set: loss_rmse: 1.8788214921951294, loss_q90: 0.87063068151474, gt_rmse: 0.2078254669904709, gt_q90: 0.3314093053340912
        # print(f"loss_rmse: {loss_rmse.mean()}, loss_q90: {loss_q90.mean()}, gt_rmse: {gt_bias_rmse.mean()}, gt_q90: {gt_bias_q90.mean()}")
        loss = w_rmse * loss_rmse + w_q90 * loss_q90  # (B,)

        return loss

class ResidualLossBiasCorrector_rmse_q90(ResidualLoss):
    """
    Residual loss function for the bias corrector.
    Predict channel-wise RMSE and Q90 of residual bias.
    Note: this loss does not apply any reduction.
    """
    def __init__(self, regression_net, bias_net, hr_mean_conditioning: bool = False, P_mean: float = 0.0, P_std: float = 1.2, sigma_data: float = 0.5):
        super().__init__(
            regression_net=regression_net,
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
            hr_mean_conditioning=hr_mean_conditioning,
        )
        self.bias_net = bias_net
        self.bias_net.eval()
        self.regression_net = regression_net
        self.regression_net.eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').to(regression_net.device)
        self.hr_mean_conditioning = hr_mean_conditioning


    def bias_correction_step(self, img_lr: torch.Tensor, img_reg: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img_reg.shape
        zero_input = torch.zeros((B, 2 * C, H, W), device=img_reg.device)
        y_conditioned = torch.cat((img_lr, img_reg), dim=1)
        
        pred_bias = self.bias_net(
            zero_input,
            y_conditioned,
            force_fp32=False,
        )

        pred_vec = F.adaptive_avg_pool2d(pred_bias, 1)      # (B, 2C, 1, 1)
        pred_vec = pred_vec.squeeze(-1).squeeze(-1)         # (B, 2C)
        pred_vec = pred_vec.view(pred_vec.shape[0], C, 2)   # (B, C, 2)

        pred_rmse = F.softplus(pred_vec[..., 0])            # (B, C)
        pred_q90 = F.softplus(pred_vec[..., 1])             # (B, C)

        return pred_rmse, pred_q90

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        time_flag: Optional[Tensor] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        B,C,H,W = img_clean.shape
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # augment for conditional generation
        y_tot = torch.cat((img_clean, img_lr), dim=1)
        augment_labels = None
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]


        # form residual
        y_mean = self.regression_net(
            torch.zeros_like(y, device=img_clean.device),
            y_lr_res,
        )
        pred_rmse, pred_q90 = self.bias_correction_step(y_lr_res, y_mean)
        if time_flag is not None:

            mask = time_flag[:, None].float()
            pred_rmse = pred_rmse * mask
            pred_q90  = pred_q90  * mask
        self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)
        # concat rmse and q90 
        pred_rmse_map = pred_rmse[:, :, None, None].expand(-1, -1, H, W)
        pred_q90_map  = pred_q90[:, :, None, None].expand(-1, -1, H, W)

        y_lr = torch.cat((y_lr, pred_rmse_map, pred_q90_map), dim=1)
        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)
        D_yn = net(
            y + n,
            y_lr,
            sigma,
            embedding_selector=None,
            global_index=(
                patching.global_index(batch_size, img_clean.device)
                if patching is not None
                else None
            ),
            augment_labels=augment_labels,
        )
        loss = weight * ((D_yn - y) ** 2)
        # lpips_loss_list = []
        
        # for channel in range(C):
        #     pred_u = D_yn[:, channel:channel + 1].repeat(1,3,1,1)
        #     targ_u = y[:, channel:channel + 1].repeat(1,3,1,1)
        #     lpips_loss = self.loss_fn_alex(pred_u, targ_u)
        #     lpips_loss_list.append(lpips_loss)
        # return torch.stack(lpips_loss_list).mean()
        return loss
class ResidualLossBiasCorrector_rmse(ResidualLossBiasCorrector_rmse_q90):
    """
    Residual loss function for the bias corrector.
    Predict channel-wise RMSE and Q90 of residual bias.
    Note: this loss does not apply any reduction.
    """
    def __init__(self, regression_net, bias_net, hr_mean_conditioning: bool = False, P_mean: float = 0.0, P_std: float = 1.2, sigma_data: float = 0.5):
        super().__init__(
            regression_net=regression_net,
            bias_net=bias_net,
            hr_mean_conditioning=hr_mean_conditioning,
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
        )

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        time_flag: Optional[Tensor] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        B,C,H,W = img_clean.shape
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # augment for conditional generation
        y_tot = torch.cat((img_clean, img_lr), dim=1)
        augment_labels = None
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]


        # form residual
        y_mean = self.regression_net(
            torch.zeros_like(y, device=img_clean.device),
            y_lr_res,
        )
        # use rmse from model 
        # pred_rmse, _ = self.bias_correction_step(y_lr_res, y_mean)
        # if time_flag is not None:
        #     mask = time_flag[:, None].float()
        #     pred_rmse = pred_rmse * mask
        # use GT rmse, but actually this has to be average over time of sample-wise rmse... may need to train it over. 
        gt_bias = img_clean - y_mean  # (B, C, H, W)

        # Channel-wise RMSE
        gt_bias_rmse = torch.sqrt((gt_bias ** 2).mean(dim=(-2, -1), keepdim=False))  # (B, C)

        self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)
        # concat rmse
        # pred_rmse_map = pred_rmse[:, :, None, None].expand(-1, -1, H, W)
        rmse_map = gt_bias_rmse[:, :, None, None].expand(-1, -1, H, W)

        # y_lr = torch.cat((y_lr, pred_rmse_map), dim=1)
        y_lr = torch.cat((y_lr, rmse_map), dim=1)
        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)
        D_yn = net(
            y + n,
            y_lr,
            sigma,
            embedding_selector=None,
            global_index=(
                patching.global_index(batch_size, img_clean.device)
                if patching is not None
                else None
            ),
            augment_labels=augment_labels,
        )
        loss = weight * ((D_yn - y) ** 2)
        # lpips_loss_list = []
        
        # for channel in range(C):
        #     pred_u = D_yn[:, channel:channel + 1].repeat(1,3,1,1)
        #     targ_u = y[:, channel:channel + 1].repeat(1,3,1,1)
        #     lpips_loss = self.loss_fn_alex(pred_u, targ_u)
        #     lpips_loss_list.append(lpips_loss)
        # return torch.stack(lpips_loss_list).mean()
        return loss
   

def bias_correction_step(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    img_reg: torch.Tensor,
    latents_shape: torch.Size,
    bias_shape: torch.Size,
) -> torch.Tensor:
    if img_reg.dim() == 4:
        B,C,H,W = img_reg.shape
    elif img_reg.dim() == 3:
        B=1
        C,H, W = img_reg.shape
    else:
        raise ValueError(f"Expected img_reg to have 3 or 4 dimensions, but found {img_reg.dim()}.")
    # Create a tensor of zeros with the given shape and move it to the appropriate device
    x_hat = torch.zeros(latents_shape, dtype=torch.float64, device=net.device)

    # Safety check: avoid silently ignoring batch elements in img_lr
    if img_lr.shape[0] > 1:
        raise ValueError(
            f"Expected img_lr to have a batch size of 1, but found {img_lr.shape[0]}."
        )

    # Perform regression on a single batch element
    img_reg_lr = downsample(img_reg[0:1, :, :, :], bias_shape)
    img_lr_lr = downsample(img_lr, bias_shape)
    img_conditioned = torch.cat((img_lr_lr, img_reg_lr), dim=1)
    print(f"img_conditioned.shape: {img_conditioned.shape}")
    with torch.inference_mode():
        x = net(x=x_hat[0:1], img_lr=img_conditioned)
    # If the batch size is greater than 1, repeat the prediction
    if B > 1:
        x = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])
    # upsampled_x = upsample(x, W, H)

    upsampled_x = downsample(x, (H,W))
    return upsampled_x

def bias_correction_step_rmse_q90(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    img_reg: torch.Tensor,
    img_gt: torch.Tensor,
    latents_shape: torch.Size,
    bias_shape: torch.Size,
) -> torch.Tensor:
    if img_reg.dim() == 4:
        B,C,H,W = img_reg.shape
    elif img_reg.dim() == 3:
        B=1
        C,H,W = img_reg.shape
    else:
        raise ValueError(f"Expected img_reg to have 3 or 4 dimensions, but found {img_reg.dim()}.")
    # Create a tensor of zeros with the given shape and move it to the appropriate device
    x_hat = torch.zeros((B, 2*C, H, W), dtype=torch.float64, device=net.device)

    # Safety check: avoid silently ignoring batch elements in img_lr
    if img_lr.shape[0] > 1:
        raise ValueError(
            f"Expected img_lr to have a batch size of 1, but found {img_lr.shape[0]}."
        )

    # Perform regression on a single batch element
    # img_reg_lr = downsample(img_reg[0:1, :, :, :], bias_shape)
    # img_lr_lr = downsample(img_lr, bias_shape)
    img_conditioned = torch.cat((img_lr, img_reg[0:1, :, :, :]), dim=1)
    
    with torch.inference_mode():
        x = net(x=x_hat[0:1], img_lr=img_conditioned)

        
    # If the batch size is greater than 1, repeat the prediction
    # if B > 1:
    #     x = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])
    gt_bias = img_gt - img_reg
    gt_bias_rmse = torch.sqrt((gt_bias ** 2).mean(dim=(-2, -1), keepdim=False))  # (B, C)
    # Channel-wise Q90 of |residual|
    gt_bias_abs = gt_bias.abs().flatten(start_dim=2)  # (B, C, H*W)
    gt_bias_q90 = torch.quantile(gt_bias_abs, q=0.9, dim=2)  # (B, C)

        
    pred_vec = F.adaptive_avg_pool2d(x, 1)    # (B, 2C, 1, 1)
    pred_vec = pred_vec.squeeze(-1).squeeze(-1)      # (B, 2C)
    pred_vec = pred_vec.view(pred_vec.shape[0], C, 2)
    pred_rmse = F.softplus(pred_vec[..., 0])            # (B, C)
    pred_q90 = F.softplus(pred_vec[..., 1])             # (B, C)

    upsampled_x = torch.zeros((B, C, H, W), device=net.device)
    upsampled_x[:,:, 0,0] = pred_rmse
    upsampled_x[:,:, 0,1] = pred_q90
    upsampled_x[:,:, 0,2] = gt_bias_rmse
    upsampled_x[:,:, 0,3] = gt_bias_q90

    return upsampled_x, pred_rmse, pred_q90



def upsample(x, H, W):
    """Extend x around edges with linear extrapolation."""
    y_shape = (
        x.shape[0],
        H,
        W,
    )
    upsample_factor = H // x.shape[1]
    y = np.empty(y_shape, dtype=np.float32)
    _zoom_extrapolate(x, y, upsample_factor)
    return y
def downsample(x, dimension):
    return F.interpolate(x, size=dimension, mode='bilinear', align_corners=False)

@jit(nopython=True)
def _zoom_extrapolate(x, y, factor):
    """Bilinear zoom with extrapolation.
    Use a numba function here because numpy/scipy options are rather slow.
    """
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1

                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)