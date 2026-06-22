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
        w_rmse: float = 1.0,
        w_q90: float = .0,
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

        err_rmse = (torch.log(pred_rmse + 1e-4) - torch.log(gt_bias_rmse + 1e-4)) ** 2
        err_q90 = (torch.log(pred_q90 + 1e-4) - torch.log(gt_bias_q90 + 1e-4)) ** 2

        loss_rmse = err_rmse.mean(dim=1)  # (B,)
        # loss_q90 = err_q90.mean(dim=1)    # (B,)
        
        # train set: loss_rmse: 5.634230613708496, loss_q90: 3.5643091201782227, gt_rmse: 0.0666382908821106, gt_q90: 0.10868420451879501
        # calibration set: loss_rmse: 1.8788214921951294, loss_q90: 0.87063068151474, gt_rmse: 0.2078254669904709, gt_q90: 0.3314093053340912
        # print(f"loss_rmse: {loss_rmse.mean()}, loss_q90: {loss_q90.mean()}, gt_rmse: {gt_bias_rmse.mean()}, gt_q90: {gt_bias_q90.mean()}")
        # loss = w_rmse * loss_rmse + w_q90 * loss_q90  # (B,)
        loss = err_rmse
        return loss
class RegressionLossBiasCorrector_quantiles:
    """
    Regression loss function for the bias corrector.
    Predict channel-wise quantiles residual bias.
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
        alpha=0.1
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
        gt_bias_flat = gt_bias.flatten(start_dim=2)

        gt_qlow  = torch.quantile(gt_bias_flat, alpha / 2, dim=2)
        gt_qhigh = torch.quantile(gt_bias_flat, 1 - alpha / 2, dim=2)

    
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

        pred_vec = F.adaptive_avg_pool2d(pred_bias, 1)
        pred_vec = pred_vec.squeeze(-1).squeeze(-1)
        pred_vec = pred_vec.view(B, C, 2)

        pred_center = pred_vec[..., 0]
        pred_width = F.softplus(pred_vec[..., 1]) + 1e-6

        pred_qlow = pred_center - pred_width
        pred_qhigh = pred_center + pred_width

        err_qlow = (pred_qlow - gt_qlow) ** 2
        err_qhigh = (pred_qhigh - gt_qhigh) ** 2

        loss = err_qlow.mean(dim=1) + err_qhigh.mean(dim=1)

        return loss

class ResidualLossBiasCorrector_quantiles(ResidualLoss):
    """
    Residual loss function for the bias corrector.
    Predict channel-wise quantiles of residual bias.
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

        pred_vec = F.adaptive_avg_pool2d(pred_bias, 1)
        pred_vec = pred_vec.squeeze(-1).squeeze(-1)
        pred_vec = pred_vec.view(B, C, 2)

        # pred_center = pred_vec[..., 0]
        # pred_width = F.softplus(pred_vec[..., 1]) + 1e-6

        # pred_qlow = pred_center - pred_width
        # pred_qhigh = pred_center + pred_width

        pred_qlow = pred_vec[...,0]
        delta = pred_vec[..., 1]
        pred_qhigh = pred_qlow + F.softplus(delta)

        return pred_qlow, pred_qhigh

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
        pred_qlow, pred_qhigh = self.bias_correction_step(y_lr_res, y_mean)
        if time_flag is not None:
            print("Using time flag")
            mask = time_flag[:, None].float()
            pred_qlow = pred_qlow * mask
            pred_qhigh  = pred_qhigh  * mask
        else:
            print("Not using time flag")
        self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)
   
        pred_qlow_map = pred_qlow[:, :, None, None].expand(-1, -1, H, W)
        pred_qhigh_map  = pred_qhigh[:, :, None, None].expand(-1, -1, H, W)

        y_lr = torch.cat((y_lr, pred_qlow_map, pred_qhigh_map), dim=1)
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
        return loss

class ResidualLossBiasCorrector_rmse(ResidualLoss):
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
    img_gt: torch.Tensor,
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

class PinballLoss():
    # from paper's open source github
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        #assert output.shape == target.shape
        
        error = output - target
        #print(f"IN PINBALL ERROR SHPAE {error.shape}")
        loss = torch.zeros_like(target, dtype=torch.float)
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

class VerifierLossQuantileScalar():
    def __init__(
        self,
        regression_net: torch.nn.Module,
        hr_mean_conditioning: bool = True,
        alpha = 0.1, # experimental default from paper
        delta = 0.1
    ):
        
        self.regression_net = regression_net
        self.regression_net.eval()
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None
        self.alpha = alpha
        self.delta = delta

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
        lead_time_label: Optional[torch.Tensor] = None,
        use_patch_grad_acc: bool = False,
    ) -> torch.Tensor:
        
        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        B,C,H,W = img_clean.shape # 64, 10, 168, 168
        # y is indexing the img_clean from the concatenated img_tot
        y = y_tot[:, : img_clean.shape[1], :, :] # GT
        # this is indexing the img_lr columns from dim 1
        y_lr = y_tot[:, img_clean.shape[1] :, :, :] # Low res ERA5
        y_lr_res = y_lr
        # if using multi-iterations of patching, switch to optimized version
        with torch.no_grad():
            if use_patch_grad_acc:
                # form residual
                if self.y_mean is None:
                    if lead_time_label is not None:
                        y_mean = self.regression_net(
                            torch.zeros_like(y, device=img_clean.device),
                            y_lr_res,
                            lead_time_label=lead_time_label,
                            augment_labels=augment_labels,
                        )
                    else:
                        y_mean = self.regression_net(
                            torch.zeros_like(y, device=img_clean.device),
                            y_lr_res,
                            augment_labels=augment_labels,
                        )
                    self.y_mean = y_mean

            # if on full domain, or if using patching without multi-iterations
            else:
                # form residual
                if lead_time_label is not None:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        lead_time_label=lead_time_label,
                        augment_labels=augment_labels,
                    )
                else:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        augment_labels=augment_labels,
                    )

            f_hat = y_mean # 64,10,168,168
        
        gt_bias = (img_clean - f_hat).view(B, C, -1)
        #print(f"GT BIAS IS {gt_bias.shape}")
        # gt_q_low  = torch.quantile(gt_bias, self.alpha, dim=2)
        # gt_q_high = torch.quantile(gt_bias, 1-self.alpha, dim=2)

        # X (LATENT STATE) NEEDS TO BE SAME AS MODEL OUTPUT, WHICH IS 2*C_OUT AND NO H,W DIMS
        x = torch.zeros(B, 2*C, H, W, device=img_clean.device) 
        #print(f"in loss, latent x shape is {x.shape}")
        if self.hr_mean_conditioning:
            y_lr_res = torch.cat((y_lr_res, f_hat), dim=1)
            
        out = net(x,
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,) # is 64,20,168,168
        # pool output together to reduce dims
        pred_vec = F.adaptive_avg_pool2d(out, 1)   # (B, 2C, 1, 1)
        pred_vec = pred_vec.squeeze(-1).squeeze(-1)  # (B, 2C)
        pred_vec = pred_vec.view(B, C, 2)            # (B, C, 2)

        q_low  = pred_vec[..., 0]
        delta  = pred_vec[..., 1]
        q_high = q_low + F.softplus(delta) # to ensure interval is nonnegative, the model essentially learns to output q_low and delta
        #print(f"TEST QUANTILES??? {q_low, q_high}")
        #print(f"SHAPES {q_low.shape}, {q_high.shape}")
        pinball_low  = PinballLoss(self.alpha / 2, reduction='none')
        pinball_high = PinballLoss(1 - self.alpha / 2, reduction='none')

        loss_low  = pinball_low(q_low.unsqueeze(-1), gt_bias)
        loss_high = pinball_high(q_high.unsqueeze(-1), gt_bias)
        # losses are now  B,C,H*W

        #print(f"METRICS\n ordering{(q_high >= q_low).float().mean()}\n asymmetry {(q_high + q_low).abs().mean()}\n accruacy {(q_low - gt_q_low).abs().mean()}, {(q_high - gt_q_high).abs().mean()}")

        return loss_low.mean(dim=2) + loss_high.mean(dim=2)
  
class VerifierQuantileScalarResidualLoss(ResidualLoss):
    """
    Residual loss function for the bias corrector.
    Predict channel-wise Q5 and Q95 of residual bias.
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

    def get_gt(self,img_clean, y_mean, alpha=0.1):
        B,C,H,W = img_clean.shape
        gt_bias = (img_clean - y_mean).view(B, C, -1) # B,C,H*W
        gt_q_low  = torch.quantile(gt_bias, alpha / 2, dim=2)
        gt_q_high = torch.quantile(gt_bias, 1-alpha/2, dim=2) # B,C
        return gt_q_low, gt_q_high

    
    def bias_correction_step(self, img_lr: torch.Tensor, img_reg: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img_reg.shape
        zero_input = torch.zeros((B, 2 * C, H, W), device=img_reg.device)
        # print(f"shape zero input : {zero_input.shape}, shape img_lr : {img_lr.shape}, shape img_reg : {img_reg.shape}")
        pred_bias = self.bias_net(zero_input,
                    img_lr, 
                    img_reg)
                    # force_fp32=True) # is 64,20,168,168
        # pool output together to reduce dims
        pred_vec = F.adaptive_avg_pool2d(pred_bias, 1)   # (B, 2C, 1, 1)
        pred_vec = pred_vec.squeeze(-1).squeeze(-1)  # (B, 2C)
        pred_vec = pred_vec.view(B, C, 2)  
    
        pred_qlow  = pred_vec[..., 0]
        delta  = pred_vec[..., 1]
        pred_qhigh = pred_qlow + F.softplus(delta)

        return pred_qlow, pred_qhigh

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        time_flag: Optional[Tensor] = None,
        alpha=0.1,
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
        # pred_qlow, pred_qhigh = self.bias_correction_step(y_lr_res, y_mean)

        # gt_bias = (img_clean - y_mean).view(B, C, -1) # B,C,H*W
        # gt_q_low  = torch.quantile(gt_bias, alpha / 2, dim=2)
        # gt_q_high = torch.quantile(gt_bias, 1-alpha/2, dim=2) # B,C
        gt_q_low, gt_q_high = self.get_gt(img_clean, y_mean, alpha)

        # if time_flag is not None:
        #     mask = time_flag[:, None].float()  # (B,1)
        #     qlow = (1 - mask) * gt_q_low + mask * pred_qlow
        #     qhigh = (1 - mask) * gt_q_high + mask * pred_qhigh

        #     # mask = time_flag[:, None].float()
        #     # qlow = pred_qlow * mask
        #     # qhigh  = pred_qhigh  * mask

        # else:
        #     # print("NOT USING TIME FLAG")
        #     qlow = pred_qlow
        #     qhigh = pred_qhigh

        qlow = gt_q_low
        qhigh=gt_q_high
        self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)
        # concat rmse and q90 
        
        pred_qlow_map = qlow[:, :, None, None].expand(-1, -1, H, W)
        pred_qhigh_map  = qhigh[:, :, None, None].expand(-1, -1, H, W)

        y_lr = torch.cat((y_lr, pred_qlow_map, pred_qhigh_map), dim=1)
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
        return loss
        # lpips_loss_list = []
        
        # for channel in range(C):
        #     pred_u = D_yn[:, channel:channel + 1].repeat(1,3,1,1)
        #     targ_u = y[:, channel:channel + 1].repeat(1,3,1,1)
        #     lpips_loss = self.loss_fn_alex(pred_u, targ_u)
        #     lpips_loss_list.append(lpips_loss)
        # return torch.stack(lpips_loss_list).mean()
 

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