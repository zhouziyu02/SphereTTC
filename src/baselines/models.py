"""Baseline backbones trained under our protocol and the model factory.

ConvLSTM, FNO, grid Transformer and ViT live here; CirT and the ClimODE-style
adapter live in external_models.py.  These backbones provide the locally trained
forecasts used by the SphereTTC evaluation protocol.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    hp = int(math.ceil(h / multiple) * multiple)
    wp = int(math.ceil(w / multiple) * multiple)
    pad_h = hp - h
    pad_w = wp - w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (h, w)


def _crop(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    return x[..., :h, :w]


class ConvLSTMCell(nn.Module):
    def __init__(self, channels: int, hidden: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden = hidden
        self.gates = nn.Conv2d(channels + hidden, 4 * hidden, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMForecast(nn.Module):
    def __init__(self, in_channels: int, out_vars: int, n_leads: int, hidden: int = 48):
        super().__init__()
        self.n_leads = n_leads
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.cell = ConvLSTMCell(hidden, hidden)
        self.head = nn.Conv2d(hidden, out_vars, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        h = torch.zeros_like(feat)
        c = torch.zeros_like(feat)
        outs = []
        step_in = feat
        for _ in range(self.n_leads):
            h, c = self.cell(step_in, (h, c))
            outs.append(self.head(h))
            step_in = h
        return torch.stack(outs, dim=1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_lat: int = 16, modes_lon: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weight_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_lat, modes_lon, 2))
        self.weight_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_lat, modes_lon, 2))

    def _compl_mul(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        w = torch.view_as_complex(weight)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        ml = min(self.modes_lat, h)
        mw = min(self.modes_lon, w // 2 + 1)
        out_ft[:, :, :ml, :mw] = self._compl_mul(x_ft[:, :, :ml, :mw], self.weight_pos[:, :, :ml, :mw])
        out_ft[:, :, -ml:, :mw] = self._compl_mul(x_ft[:, :, -ml:, :mw], self.weight_neg[:, :, :ml, :mw])
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOForecast(nn.Module):
    def __init__(self, in_channels: int, out_vars: int, n_leads: int, width: int = 48, layers: int = 4):
        super().__init__()
        self.n_leads = n_leads
        self.out_vars = out_vars
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, n_leads * out_vars, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for spec, local in zip(self.spectral, self.local):
            h = F.gelu(spec(h) + local(h))
        y = self.proj(h)
        b, _, lat, lon = y.shape
        return y.view(b, self.n_leads, self.out_vars, lat, lon)


class GridTransformerForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_vars: int,
        n_leads: int,
        hidden: int = 96,
        stride: int = 8,
        layers: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.out_vars = out_vars
        self.stride = stride
        self.down = nn.Conv2d(in_channels, hidden, kernel_size=stride, stride=stride)
        block = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 4, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Conv2d(hidden, n_leads * out_vars, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, size = _pad_to_multiple(x, self.stride)
        tokens = self.down(x)
        b, c, hp, wp = tokens.shape
        seq = tokens.flatten(2).transpose(1, 2)
        seq = self.encoder(seq)
        low = seq.transpose(1, 2).view(b, c, hp, wp)
        high = F.interpolate(self.head(low), size=x.shape[-2:], mode="bilinear", align_corners=False)
        high = _crop(high, size)
        return high.view(b, self.n_leads, self.out_vars, *size)


class ViTForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_vars: int,
        n_leads: int,
        hidden: int = 128,
        patch_size: int = 16,
        layers: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.n_leads = n_leads
        self.out_vars = out_vars
        self.patch_size = patch_size
        self.patch = nn.Conv2d(in_channels, hidden, kernel_size=patch_size, stride=patch_size)
        block = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 4, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.unpatch = nn.Linear(hidden, n_leads * out_vars * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, size = _pad_to_multiple(x, self.patch_size)
        tokens = self.patch(x)
        b, c, hp, wp = tokens.shape
        seq = tokens.flatten(2).transpose(1, 2)
        seq = self.encoder(seq)
        patches = self.unpatch(seq)
        p = self.patch_size
        patches = patches.view(b, hp, wp, self.n_leads * self.out_vars, p, p)
        y = patches.permute(0, 3, 1, 4, 2, 5).reshape(b, self.n_leads * self.out_vars, hp * p, wp * p)
        y = _crop(y, size)
        return y.view(b, self.n_leads, self.out_vars, *size)


def build_model(name: str, in_channels: int, out_vars: int, n_leads: int) -> nn.Module:
    name = name.lower()
    if name == "convlstm":
        return ConvLSTMForecast(in_channels, out_vars, n_leads)
    if name == "fno":
        return FNOForecast(in_channels, out_vars, n_leads)
    if name == "transformer":
        return GridTransformerForecast(in_channels, out_vars, n_leads)
    if name == "vit":
        return ViTForecast(in_channels, out_vars, n_leads)
    if name == "cirt":
        from .external_models import CirTForecast

        return CirTForecast(in_channels, out_vars, n_leads)
    if name == "climode":
        from .external_models import ClimODEForecast

        return ClimODEForecast(in_channels, out_vars, n_leads)
    raise ValueError(f"Unknown baseline model: {name}")
