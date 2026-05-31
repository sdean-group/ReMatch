# uses regular MSE loss
import torch
import torch.nn.functional as F
import torch.nn as nn

class ConvFNOLoss():
    def __init__(self):
        """
        Arguments
        ----------
        """
        return
    
    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
    ) -> torch.Tensor:

        y_hat = net(img_lr)
        #print(f"DSLKFJSLFJSDLKFJDSKFJ {img_lr.shape}, {img_clean.shape}, {y_hat.shape}")

        return F.mse_loss(y_hat, img_clean)
    
def convfno_step(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    latents_shape: torch.Size,
) -> torch.Tensor:
    # remove latent shpae and xhat
    
    # Create a tensor of zeros with the given shape and move it to the appropriate device
    x_hat = torch.zeros(latents_shape, dtype=torch.float64, device=net.device)

    # Safety check: avoid silently ignoring batch elements in img_lr
    if img_lr.shape[0] > 1:
        raise ValueError(
            f"Expected img_lr to have a batch size of 1, but found {img_lr.shape[0]}."
        )

    # Perform regression on a single batch element
    with torch.inference_mode():
            x = net(img_lr)

    # If the batch size is greater than 1, repeat the prediction
    if x_hat.shape[0] > 1:
        x = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

    return x

