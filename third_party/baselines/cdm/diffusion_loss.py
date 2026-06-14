import torch
from torch import Tensor
from physicsnemo.metrics.diffusion import ResidualLoss
from typing import Callable, Optional, Tuple
# from physicsnemo.models.diffusion.patching import RandomPatching2D

class DiffusionLoss(ResidualLoss):

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
    ) -> Tensor:
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
        y_tot = torch.cat((img_clean, img_lr), dim=1)
  
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        self.y_mean = torch.zeros_like(img_clean)


        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)

        D_yn = net(
            y + n,
            y_lr,
            sigma,
        )
        loss = weight * ((D_yn - y) ** 2)

        return loss
