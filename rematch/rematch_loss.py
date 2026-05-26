from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from physicsnemo.utils.patching import RandomPatching2D

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


class ResidualLoss_swinir:
    """
    Mixture loss function for denoising score matching.

    This class implements a loss function that combines deterministic
    regression with denoising score matching. It uses a pre-trained regression
    network to compute residuals before applying the diffusion process.

    Parameters
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.
            Expected signature: `net(zero_input, y_lr,
            lead_time_label=lead_time_label, augment_labels=augment_labels)` or
            `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
                zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                lead_time_label (torch.Tensor, optional): Optional lead time labels
                augment_labels (torch.Tensor, optional): Optional augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning predicted, by default False.
            When True, the mean prediction from `regression_net` is channel-wise
            concatenated with `img_lr` for conditioning.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None

    def get_noise_params(self, y: Tensor) -> Tensor:
        """
        Compute the noise parameters to apply denoising score matching.

        Parameters
        ----------
        y : torch.Tensor
            Latent state of shape :math:`(B, *)`. Only used to determine the shape of
            the noise and create tensors on the same device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
            - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(y) * sigma
        return n, sigma, weight

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        img_lr_raw: Tensor,
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
        y_lr_res = y_lr
        batch_size = y.shape[0]
        y_mean = self.regression_net(img_lr_raw)
        
        self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

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
