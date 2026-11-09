import math

def cosine_guidance_weights(num_steps, w_start=0.0, w_end=8.0, t_max=None):
    """PyTorch CosineAnnealingLR-style schedule: w_start at i=0, w_end at i=t_max."""
    t_max = (num_steps - 1) if t_max is None else t_max
    return [
        w_end + (w_start - w_end) * 0.5 * (1 + math.cos(math.pi * i / t_max))
        for i in range(num_steps)
    ]


def cfg_stochastic_sampler(
    net: torch.nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    class_labels: Optional[Tensor] = None,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    patching: Optional[GridPatching2D] = None,
    mean_hr: Optional[Tensor] = None,
    lead_time_label: Optional[Tensor] = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
    # --- CFG arguments ---
    guidance_w_min: float = 0.0,
    guidance_w_max: float = 8.0,
    guidance_reverse: bool = False,
    uncond_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    r"""
    EDM stochastic sampler with classifier-free guidance (CFG).

    Identical to ``stochastic_sampler`` except each denoiser evaluation is
    replaced by ``D_cond + w_i * (D_cond - D_uncond)``, where ``w_i`` follows a
    cosine schedule from ``guidance_w_min`` to ``guidance_w_max`` over the
    ``num_steps`` sampling steps.

    Extra parameters
    ----------------
    guidance_w_min, guidance_w_max : float
        Endpoints of the cosine guidance schedule (defaults 0 -> 8).
    guidance_reverse : bool
        If True, flip the schedule (strong guidance first, decaying).
    uncond_fn : Callable[[Tensor], Tensor], optional
        Maps the (possibly patched) conditioning tensor ``x_lr`` to its
        "null" version. Must match how condition dropout was done in
        training. Defaults to ``torch.zeros_like``.
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if patching is not None and not isinstance(patching, GridPatching2D):
        raise ValueError("patching must be an instance of GridPatching2D.")
    if patching:
        if img_lr.shape[-2:] != latents.shape[-2:]:
            raise ValueError(
                f"img_lr and latents must have the same height and width, "
                f"but found {img_lr.shape[-2:]} vs {latents.shape[-2:]}. "
            )
    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )

    # Time step discretization.
    step_indices = torch.arange(num_steps, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # CFG weights, one per step
    guidance_ws = cosine_guidance_weights(
        18, w_start=0.0, w_end=8.0
    )
    if uncond_fn is None:
        uncond_fn = torch.zeros_like

    batch_size = img_lr.shape[0]

    # conditioning = [mean_hr, img_lr, global_lr, pos_embd]
    x_lr = img_lr
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"mean_hr and img_lr must have the same height and width, "
                f"but found {mean_hr.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)

    if patching:
        x_lr = _apply_wrapper_Cin_channels(
            patching=patching, input=x_lr, additional_input=img_lr
        )

        def patch_embedding_selector(emb):
            return patching.apply(emb.expand(batch_size, -1, -1, -1))
    else:
        patch_embedding_selector = None

    optional_args = {}
    if lead_time_label is not None:
        optional_args["lead_time_label"] = lead_time_label
    if patching:
        optional_args["embedding_selector"] = patch_embedding_selector

    x_lr = x_lr.to(latents.device)
    x_lr_null = uncond_fn(x_lr)  # computed once; reused every step

    def _call_net(x_batch, t, cond):
        if isinstance(net, EDMPrecond):
            return net(
                x_batch, t, condition=cond, class_labels=class_labels, **optional_args
            )
        return net(x_batch, cond, t, class_labels, **optional_args)

    def guided_denoise(x, t, w):
        """CFG-combined denoiser on the full (un-patched) image."""
        x_batch = _apply_wrapper_select(input=x, patching=patching)(
            patching=patching, input=x
        ).to(latents.device)

        d_cond = _call_net(x_batch, t, x_lr)
        if w != 0:
            d_uncond = _call_net(x_batch, t, x_lr_null)
            denoised = d_cond + w * (d_cond - d_uncond)
        else:
            denoised = d_cond

        if patching:
            denoised = _fuse_wrapper(
                patching=patching, input=denoised, batch_size=batch_size
            )
        return denoised

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        w = guidance_ws[i]

        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step
        denoised = guided_denoise(x_hat, t_hat, w)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction (same w as this step)
        if i < num_steps - 1:
            denoised = guided_denoise(x_next, t_next, w)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
