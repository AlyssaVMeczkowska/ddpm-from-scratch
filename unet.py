"""
U-Net Denoising Network
Implements the U-Net architecture used in DDPM with:
- Sinusoidal timestep embeddings
- Residual blocks with group normalization
- Multi-head self-attention at bottleneck
- Skip connections between encoder and decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for timestep conditioning.
    Transforms scalar timestep t into a rich embedding vector,
    same idea as positional encodings in transformers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class WeightStandardizedConv2d(nn.Conv2d):
    """
    Conv2d with weight standardization.
    Works better than BatchNorm with GroupNorm.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        std = weight.std(dim=[1, 2, 3], keepdim=True) + 1e-5
        weight = (weight - mean) / std
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class Block(nn.Module):
    """Conv → GroupNorm → SiLU block with optional scale-shift conditioning."""
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning via scale-shift.
    Core building block of the U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2))
            if time_emb_dim is not None else None
        )
        self.block1 = Block(in_channels, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class MultiHeadAttention(nn.Module):
    """Self-attention for spatial feature maps."""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(1, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = [t.reshape(b, self.num_heads, c // self.num_heads, h * w) for t in qkv]

        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(b, c, h, w)
        return self.to_out(out) + x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net for DDPM noise prediction.

    Architecture:
      Encoder: series of ResNet blocks + downsampling
      Bottleneck: ResNet + Self-Attention + ResNet
      Decoder: series of ResNet blocks (with skip connections) + upsampling
    """
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (2,),
        dropout: float = 0.0,
    ):
        super().__init__()

        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, 7, padding=3)

        # Encoder
        self.downs = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels

        for level, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResnetBlock(now_channels, out_channels, time_emb_dim))
                now_channels = out_channels
                channels.append(now_channels)
                if level + 1 in attention_resolutions:
                    self.downs.append(MultiHeadAttention(now_channels))

            if level < len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # Bottleneck
        self.mid_block1 = ResnetBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = MultiHeadAttention(now_channels)
        self.mid_block2 = ResnetBlock(now_channels, now_channels, time_emb_dim)

        # Decoder
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = model_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResnetBlock(channels.pop() + now_channels, out_channels, time_emb_dim))
                now_channels = out_channels
                if level + 1 in attention_resolutions:
                    self.ups.append(MultiHeadAttention(now_channels))

            if level > 0:
                self.ups.append(Upsample(now_channels))

        # Output
        self.out_norm = nn.GroupNorm(8, now_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(now_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = [x]

        # Encode
        for layer in self.downs:
            if isinstance(layer, (ResnetBlock,)):
                x = layer(x, time_emb)
            else:
                x = layer(x)
            if not isinstance(layer, MultiHeadAttention):
                skips.append(x)

        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # Decode
        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                x = torch.cat((x, skips.pop()), dim=1)
                x = layer(x, time_emb)
            else:
                x = layer(x)

        x = self.out_norm(x)
        x = self.out_act(x)
        return self.out_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
