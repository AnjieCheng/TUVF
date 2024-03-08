__all__ = [
    "CIPSskip",
    "CIPSres",
]

import math

import torch
from torch import nn
import torch.nn.functional as F
from src.training.training_utils import *

from src.training.networks_cips_block import FullyConnectedLayer, ConstantInput3D, ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock


class CIPSskip(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01, activation=None, channel_multiplier=2, **kwargs):
        super(CIPSskip, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(hidden_size)
        self.emb = ConstantInput(hidden_size, size=size)

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 32 * channel_multiplier,
            8: 16 * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.conv1 = StyledConv(int(multiplier * hidden_size), in_channels, 1, style_dim, demodulate=demodulate, activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim, demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim, demodulate=demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channels, style_dim, upsample=False))

            in_channels = out_channels

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))

        self.style = nn.Sequential(*layers)

    def forward(
        self,
        coords,
        latent,
        return_latents=False,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        latent = latent[0]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)

        x = self.lff(coords)

        batch_size, _, w, h = coords.shape
        if self.training and w == h == self.size:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode="border",
                mode="bilinear",
            )

        x = torch.cat([x, emb], 1)

        rgb = 0

        x = self.conv1(x, latent)
        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.linears[i * self.to_rgb_stride + j](x, latent)

            rgb = self.to_rgbs[i](x, latent, rgb)

        if return_latents:
            return rgb, latent
        else:
            return rgb, None


class CIPSres(nn.Module):
    def __init__(self, size=256, hidden_size=256, n_mlp=8, style_dim=512, lr_mlp=0.01, is_emb=False, geo_cond=False, g_dim=256, activation=None, channel_multiplier=2, out_dim=32, **kwargs):
        super(CIPSres, self).__init__()
        self.geo_cond = geo_cond
        if geo_cond:
            self.geo_embed = FullyConnectedLayer(g_dim, style_dim)

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.is_emb = is_emb
        if is_emb:
            self.lff = LFF(int(hidden_size))
            self.emb = ConstantInput3D(hidden_size, size=64)
        else:
            self.lff = LFF(int(hidden_size) * 2)
            self.emb = None
        self.num_ws = 1

        self.channels = {
            0: 256,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: out_dim,
        }

        self.linears = nn.ModuleList()
        in_channels = 512  # int(self.channels[0])
        multiplier = 2
        self.linears.append(StyledConv(int(multiplier * hidden_size), in_channels, 1, style_dim, demodulate=demodulate, activation=activation))

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        for i in range(0, self.log_size):
            out_channels = self.channels[i]
            self.linears.append(StyledResBlock(in_channels, out_channels, 1, style_dim, demodulate=demodulate, activation=activation))
            in_channels = out_channels

        # self.to_rgb_last = ToRGB(in_channels, style_dim, upsample=False)

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            if i == 0 and geo_cond:
                in_dim = style_dim * 2
            else:
                in_dim = style_dim

            layers.append(EqualLinear(in_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))

        self.style = nn.Sequential(*layers)

    def forward(
        self,
        coords,
        latent,
        geo_cond=None,
        return_latents=False,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        latent = latent[0]
        coords_normed = coords / 0.5
        coords_normed = coords_normed.transpose(1, 2)[:, :, :, None]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if self.geo_cond:
            assert geo_cond is not None
            geo_cond = geo_cond[0]
            geo_cond = self.geo_embed(geo_cond)
            latent = torch.cat([latent, geo_cond], dim=-1)

        if not input_is_latent:
            latent = self.style(latent)

        x = self.lff(coords_normed)

        batch_size, _, w, h = coords_normed.shape

        if self.is_emb:
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1, -1),
                coords_normed.permute(0, 2, 3, 1).contiguous().view(batch_size, w, h, 1, 3),
                padding_mode="border",
                mode="bilinear",
            ).squeeze(4)

            out = torch.cat([x, emb], 1)

        else:
            # out = torch.cat([x, coords_normed], 1)
            out = x

        for con in self.linears:
            out = con(out, latent)

        # out = self.to_rgb_last(out, latent)
        out = out.transpose(1, 2).squeeze(3)

        # out = out*0 + torch.clip((coords+0.5), 0, 1) # normalize color to 0-1

        if return_latents:
            return out, latent
        else:
            return out


class CIPS2D(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01, activation=None, channel_multiplier=2, **kwargs):
        super(CIPS2D, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.lff = LFF(int(hidden_size), dim=2)
        self.emb = ConstantInput(hidden_size, size=size)
        self.num_ws = 1

        self.channels = {
            0: 512,
            1: 512,
            2: 512,
            3: 512,
            4: 256 * channel_multiplier,
            5: 128 * channel_multiplier,
            6: 64 * channel_multiplier,
            7: 64 * channel_multiplier,
            8: 32 * channel_multiplier,
        }

        self.linears = nn.ModuleList()
        in_channels = int(self.channels[0])
        multiplier = 2
        self.linears.append(StyledConv(int(multiplier * hidden_size), in_channels, 1, style_dim, demodulate=demodulate, activation=activation))

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledResBlock(in_channels, out_channels, 1, style_dim, demodulate=demodulate, activation=activation))
            in_channels = out_channels

        self.to_rgb_last = ToRGB(in_channels, style_dim, upsample=False, out_dim=32)

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))

        self.style = nn.Sequential(*layers)

    def forward(
        self,
        coords,
        latent,
        return_latents=False,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        latent = latent[0]

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if not input_is_latent:
            latent = self.style(latent)

        x = self.lff(coords)

        batch_size, _, w, h = coords.shape
        if self.training and w == h == self.size:
            emb = self.emb(x)
        else:
            emb = F.grid_sample(
                self.emb.input.expand(batch_size, -1, -1, -1),
                coords.permute(0, 2, 3, 1).contiguous(),
                padding_mode="border",
                mode="bilinear",
            )
        out = torch.cat([x, emb], 1)

        for con in self.linears:
            out = con(out, latent)

        out = self.to_rgb_last(out, latent)

        if return_latents:
            return out, latent
        else:
            return out
