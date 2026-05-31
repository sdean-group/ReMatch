import torch
from torch import Tensor
from physicsnemo.metrics.diffusion import ResidualLoss
from typing import Callable, Optional, Tuple
from physicsnemo.models.diffusion.patching import RandomPatching2D


class GuidanceFreeResidualLoss(ResidualLoss):
    def __init__(
        self,
        regression_net: torch.nn.Module,
        p_conditional_drop=0.10,
        **residual_loss_kwargs,
    ):
            super().__init__(regression_net=regression_net, **residual_loss_kwargs)
            self.p_conditional_drop = p_conditional_drop
    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
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
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        # TRUTH
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # if using multi-iterations of patching, switch to optimized version
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

            self.y_mean = y_mean
       
        y = y - self.y_mean
        

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1) 

        # drop condition 10% of the time
        drop_mask = (torch.rand(batch_size, device=y.device) < self.p_conditional_drop)

        # expand mask to match spatial dims
        drop_mask = drop_mask.view(batch_size, 1, 1, 1)

        # define "null conditioning"
        null_cond = torch.zeros_like(y_lr)  # TODO: could try learned embedding

        # apply per-sample masking
        y_lr = torch.where(drop_mask, null_cond, y_lr)

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        # removed patch_embedding_selector due to compilation issue with dynamo.
        if patching:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y_patched = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)

            y = y_patched
            y_lr = y_lr_patched

        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)

        if lead_time_label is not None:
            D_yn = net(
                y+n,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            # stronger conditioning makes diversity die
            D_yn = net(
                y+n,
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
    
class VerifierMoranGuidanceFreeResidualLoss(ResidualLoss):
    def __init__(self, regression_net, bias_net, hr_mean_conditioning: bool = False, P_mean: float = 0.0, P_std: float = 1.2, sigma_data: float = 0.5, p_conditional_drop=0.10):
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
        self.hr_mean_conditioning = hr_mean_conditioning
        self.p_conditional_drop = p_conditional_drop
    
    def verifier_moran_pixel_residual_step(
        self,
        img_lr: torch.Tensor,
        img_reg: torch.Tensor,
    ) -> torch.Tensor:
        
        # Create a tensor of zeros with the given shape and move it to the appropriate 
        B, C, H, W = img_reg.shape
        # NOTE: kind of hardcoding this rn
        zero_input = torch.zeros_like(img_reg)

        y_lr = torch.cat((img_lr, img_reg), dim=1)
        with torch.no_grad():
            pred_bias = self.bias_net(zero_input, y_lr, force_fp32=False) # 64,10,168,168 (matches img_lr)
    
        return pred_bias

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
        with torch.no_grad():
            y_mean = self.regression_net(
                torch.zeros_like(y, device=img_clean.device),
                y_lr_res,
            )
            pred_moran = self.verifier_moran_pixel_residual_step(y_lr_res, y_mean)

        if time_flag is not None:
            mask = time_flag[:, None].float()  # (B,1)
            mask = mask.view(B,1,1,1)
            pred_moran_map = mask * pred_moran # 64, 10, 168, 168
        else:
            print("NOT USING TIME FLAG")
            pred_moran_map = pred_moran

        self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)



        # drop condition 10% of the time
        drop_mask = (torch.rand(batch_size, device=y.device) < self.p_conditional_drop)

        # expand mask to match spatial dims
        drop_mask = drop_mask.view(batch_size, 1, 1, 1)

        # define "null conditioning"
        moran_cond = torch.where(drop_mask, torch.zeros_like(pred_moran_map), pred_moran_map)

        y_lr = torch.cat((y_lr, moran_cond), dim=1)

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
    
