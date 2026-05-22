# swinir_2d_sr.py
# Minimal SwinIR for 2D multi-channel super-resolution
# Target use:
#   LR: (B, 10, 21, 21)
#   HR: (B, 10, 168, 168)

from __future__ import annotations

import math
from typing import Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)

    Returns:
        windows: (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(
            f"H and W must be divisible by window_size. "
            f"Got H={H}, W={W}, window_size={window_size}."
        )

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Args:
        windows: (num_windows * B, window_size, window_size, C)

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, -1)
    )
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        num_relative_positions = (
            (2 * window_size[0] - 1)
            * (2 * window_size[1] - 1)
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        # indexing='ij' avoids PyTorch warning in newer versions.
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (num_windows * B, N, C)
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (
                attn.view(B_ // nW, nW, self.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)
            )
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        H, W = input_resolution
        if min(H, W) <= window_size:
            window_size = min(H, W)
            shift_size = 0

        if not (0 <= shift_size < window_size):
            raise ValueError(
                f"shift_size must satisfy 0 <= shift_size < window_size. "
                f"Got shift_size={shift_size}, window_size={window_size}."
            )

        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: Tuple[int, int]) -> torch.Tensor:
        H, W = x_size

        img_mask = torch.zeros((1, H, W, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
    ) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        if L != H * W:
            raise ValueError(
                f"Input feature has wrong size. "
                f"Got L={L}, but H*W={H*W}."
            )

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C,
        )

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            mask = self.calculate_mask(x_size).to(x.device)
            attn_windows = self.attn(x_windows, mask=mask)

        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C,
        )

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_drop_path = (
                drop_path[i]
                if isinstance(drop_path, (list, tuple))
                else drop_path
            )

            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=block_drop_path,
                    norm_layer=norm_layer,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
    ) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        return x


class PatchEmbed(nn.Module):
    """
    Here patch_size is kept for compatibility with SwinIR,
    but for this SR model patch_size should be 1.
    """

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 21,
        patch_size: int | Tuple[int, int] = 1,
        embed_dim: int = 60,
        norm_layer: type[nn.Module] | None = nn.LayerNorm,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if patch_size != (1, 1):
            raise ValueError("This minimal SwinIR implementation assumes patch_size=1.")

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.num_patches = img_size[0] * img_size[1]
        self.embed_dim = embed_dim

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim: int = 60):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, HW, C = x.shape
        H, W = x_size

        if HW != H * W:
            raise ValueError(
                f"Token number mismatch. Got HW={HW}, expected H*W={H*W}."
            )

        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        resi_connection: str = "1conv",
    ):
        super().__init__()

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=input_resolution,
            patch_size=1,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
    ) -> torch.Tensor:
        res = self.residual_group(x, x_size)
        res = self.patch_unembed(res, x_size)
        res = self.conv(res)
        res = self.patch_embed(res)
        return res + x


class UpsampleOneStep(nn.Sequential):
    """
    Lightweight one-step PixelShuffle upsampling.

    Input:
        (B, num_feat, H, W)

    Output:
        (B, num_out_ch, scale*H, scale*W)
    """

    def __init__(
        self,
        scale: int,
        num_feat: int,
        num_out_ch: int,
    ):
        if scale < 1:
            raise ValueError(f"scale must be >= 1. Got {scale}.")

        layers = [
            nn.Conv2d(
                num_feat,
                (scale ** 2) * num_out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(scale),
        ]

        super().__init__(*layers)

class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int):
        m = []

        if scale in [2, 4, 8]:
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale: {scale}")

        super().__init__(*m)
class SwinIR2DSR(nn.Module):
    """
    Minimal SwinIR for 2D multi-channel super-resolution.

    Default setting:
        input : (B, 10, 21, 21)
        output: (B, 10, 168, 168)
    """

    def __init__(
        self,
        img_size: int | Tuple[int, int] = (21, 21),
        in_chans: int = 10,
        out_chans: int = 10,
        upscale: int = 8,
        embed_dim: int = 60,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        resi_connection: str = "1conv",
        strict_img_size: bool = False,
        multi_step_sampler: bool = False,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)

        if len(depths) != len(num_heads):
            raise ValueError(
                f"depths and num_heads must have same length. "
                f"Got len(depths)={len(depths)}, len(num_heads)={len(num_heads)}."
            )

        if embed_dim % min(num_heads) != 0:
            raise ValueError(
                f"embed_dim={embed_dim} should be divisible by all num_heads={num_heads}."
            )

        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.window_size = window_size
        self.strict_img_size = strict_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.multi_step_sampler = multi_step_sampler

        self.conv_first = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=1,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        self.layers = nn.ModuleList()
        depth_cursor = 0

        for i_layer in range(self.num_layers):
            depth_i = depths[i_layer]
            layer_drop_paths = dpr[depth_cursor: depth_cursor + depth_i]
            depth_cursor += depth_i

            layer = RSTB(
                dim=embed_dim,
                input_resolution=img_size,
                depth=depth_i,
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=layer_drop_paths,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unknown resi_connection: {resi_connection}")

        if self.multi_step_sampler:
            self.num_feat = 64

            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, self.num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )

            self.upsample = Upsample(upscale, self.num_feat)

            self.conv_last = nn.Conv2d(
                self.num_feat,
                out_chans,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        else:
            self.upsample = UpsampleOneStep(
                scale=upscale,
                num_feat=embed_dim,
                num_out_ch=out_chans,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        if self.strict_img_size and (h, w) != self.img_size:
            raise ValueError(
                f"Expected input spatial size {self.img_size}, "
                f"but got {(h, w)}."
            )

        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if mod_pad_h == 0 and mod_pad_w == 0:
            return x

        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 10, 21, 21)

        Returns:
            y: (B, 10, 168, 168)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}.")

        B, C, H, W = x.shape

        if C != self.in_chans:
            raise ValueError(
                f"Expected {self.in_chans} input channels, got {C}."
            )

        x = self.check_image_size(x)

        x_first = self.conv_first(x)
        x_body = self.forward_features(x_first)
        x_body = self.conv_after_body(x_body) + x_first
        if self.multi_step_sampler:
            out = self.conv_before_upsample(x_body)
            out = self.upsample(out)
            out = self.conv_last(out)
        else:
            out = self.upsample(x_body)

        return out[:, :, : H * self.upscale, : W * self.upscale]


if __name__ == "__main__":
    model = SwinIR2DSR(
        img_size=(21, 21),
        in_chans=14,
        out_chans=10,
        upscale=8,
        embed_dim=60,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=7,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        resi_connection="1conv",
        use_checkpoint=False,
    )

    x = torch.randn(2, 10, 21, 21)
    y = model(x)

    print("input :", x.shape)
    print("output:", y.shape)

    assert y.shape == (2, 10, 168, 168)