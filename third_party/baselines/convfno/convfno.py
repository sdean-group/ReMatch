from typing import List

from physicsnemo.models.fno import FNO, FNO2DEncoder
# from physicsnemo.nn.spectral_layers import SpectralConv2d
from physicsnemo.models.layers.spectral_layers import SpectralConv2d
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .rcan import Upsampler, default_conv
from . import tools

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i ==0: 
                tools.initialize_weights(m[-1], 'relu', None, 0.1)
            else:
                tools.initialize_weights(m[-1], 'linear', None, 0.1)
            if bn:
                # modified from 3d norm to 2d norm?
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
            

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class UpsamplerDecoder(nn.Module):
    def __init__(self,n_feats,out_channel,kernel_size,scale, conv=default_conv):
        super(UpsamplerDecoder, self).__init__()

        
        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, out_channel, kernel_size)
        ]
        tools.initialize_weights(m_tail[-1], 'linear', None, 0.1)

        self.tail = nn.Sequential(*m_tail)
        self.upscale = scale

    def forward(self, x):
        x = self.tail(x)
        return x 
    
class FNO2DResidualEncoder(FNO2DEncoder):
    def build_fno(self, num_fno_modes: List[int]) -> None:
        """construct FNO block.
        Parameters
        ----------
        num_fno_modes : List[int]
            Number of Fourier modes kept in spectral convolutions

        """
        # Build Neural Fourier Operators
        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))
            # NOTE: why 3 for kernel size?
            self.res_layers.append(ResBlock(default_conv, self.fno_width,3,res_scale=0.1))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"
            )

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
        x = self.lift_network(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            x_fp32 = x.float()
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x_fp32) + w(x_fp32) + self.res_layers[k](x_fp32))
            else:
                x = conv(x_fp32) + w(x_fp32) + self.res_layers[k](x_fp32)

        # remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

class ConvFNO(FNO):
    def __init__(self,
                nfeats,
                out_chans,
               kernel_size, upscale=8,**fno_kwargs):
        super().__init__(**fno_kwargs)
        self.spec_encoder = FNO2DResidualEncoder(
            in_channels=fno_kwargs["in_channels"],
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=nfeats,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
        )
        self.decoder_net = UpsamplerDecoder(nfeats,fno_kwargs["out_channels"],kernel_size,upscale)
    def forward(self, x):
        x = self.spec_encoder(x)
        x = self.decoder_net(x)
        return x