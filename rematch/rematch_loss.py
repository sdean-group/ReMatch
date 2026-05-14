from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from physicsnemo.models.diffusion.patching import RandomPatching2D

from physicsnemo.metrics.diffusion import ResidualLoss



class ResidualLoss_preload_loss(ResidualLoss):
    """
    Residual loss function for the bias corrector.
    Predict channel-wise RMSE and Q90 of residual bias.
    Note: this loss does not apply any reduction.
    """
    def __init__(self, hr_mean_conditioning: bool = False, P_mean: float = 0.0, P_std: float = 1.2, sigma_data: float = 0.5):
        super().__init__(
            regression_net=None,
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
            hr_mean_conditioning=hr_mean_conditioning,
        )
        self.hr_mean_conditioning = hr_mean_conditioning
        self.idx_cnt=0

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        img_reg: Tensor,
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
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        batch_size = y.shape[0]

        self.y_mean = img_reg.to(y.device)
        self.idx_cnt += 1
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)
            

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

