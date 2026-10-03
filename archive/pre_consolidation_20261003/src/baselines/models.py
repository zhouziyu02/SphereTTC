from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_harmonics import InverseRealSHT, RealSHT


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


def _spherical_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Periodically pad longitude and replicate the polar latitude boundary."""
    if pad <= 0:
        return x
    x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    return F.pad(x, (0, 0, pad, pad), mode="replicate")


class SphericalDepthwiseBlock(nn.Module):
    """A local residual block without a false periodic latitude boundary."""

    def __init__(self, channels: int, expansion: int = 2, kernel_size: int = 5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=0,
            groups=channels,
        )
        hidden = expansion * channels
        self.expand = nn.Conv2d(channels, 2 * hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(_spherical_pad(self.norm(x), self.pad))
        value, gate = self.expand(y).chunk(2, dim=1)
        return x + self.project(F.gelu(value) * torch.sigmoid(gate))


class StableSphericalBandDynamics(nn.Module):
    """Low-rank lead-conditioned dynamics in a truncated spherical basis.

    Each latent spherical-frequency band evolves with a contractive complex
    semigroup.  High-frequency coefficients outside the truncation are
    retained by the persistence path in :class:`SphereDynForecast`.
    """

    def __init__(
        self,
        channels: int,
        n_leads: int,
        nlat: int,
        nlon: int,
        rank: int = 24,
        lmax: int = 48,
        mmax: int = 48,
        n_bands: int = 6,
        stable: bool = True,
        bounded: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_leads = int(n_leads)
        self.rank = int(rank)
        self.lmax = min(int(lmax), int(nlat))
        self.mmax = min(int(mmax), int(nlon) // 2 + 1, self.lmax)
        self.stable = bool(stable)
        self.bounded = bool(bounded)
        if self.stable and self.bounded:
            raise ValueError("stable and bounded dynamics are mutually exclusive")
        self.sht = RealSHT(
            nlat,
            nlon,
            lmax=self.lmax,
            mmax=self.mmax,
            grid="equiangular",
        )
        self.isht = InverseRealSHT(
            nlat,
            nlon,
            lmax=self.lmax,
            mmax=self.mmax,
            grid="equiangular",
        )
        scale = 1.0 / math.sqrt(max(1, channels))
        self.encode_weight = nn.Parameter(
            scale * torch.randn(rank, channels)
        )
        self.decode_weight = nn.Parameter(
            0.02 * torch.randn(channels, rank)
        )
        n_bands = max(1, min(int(n_bands), self.lmax))
        band_index = torch.div(
            torch.arange(self.lmax) * n_bands,
            self.lmax,
            rounding_mode="floor",
        ).clamp_max(n_bands - 1)
        self.register_buffer("band_index", band_index, persistent=True)
        if stable:
            self.log_damping = nn.Parameter(
                torch.full((rank, n_bands), -4.0)
            )
            self.phase = nn.Parameter(torch.zeros(rank, n_bands))
            self.amplitude = nn.Parameter(torch.zeros(rank, n_bands))
        elif bounded:
            # Per-lead dynamics are flexible, but the spectral response cannot
            # explode: exp(±0.4) bounds every modal magnitude to [0.67, 1.49].
            self.bounded_log_magnitude = nn.Parameter(
                torch.zeros(n_leads, rank, n_bands)
            )
            self.bounded_phase = nn.Parameter(
                torch.zeros(n_leads, rank, n_bands)
            )
        elif not stable:
            self.unconstrained_real = nn.Parameter(
                torch.zeros(n_leads, rank, n_bands)
            )
            self.unconstrained_imag = nn.Parameter(
                torch.zeros(n_leads, rank, n_bands)
            )

    def _lead_multiplier(
        self,
        lead_days: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.stable:
            time = lead_days.to(device=device, dtype=torch.float32)
            time = time / time.max().clamp_min(1.0)
            damping = F.softplus(self.log_damping)
            magnitude = (
                1.0 + 0.5 * torch.tanh(self.amplitude)
            )[None] * torch.exp(-time[:, None, None] * damping[None])
            angle = (
                math.pi
                * time[:, None, None]
                * torch.tanh(self.phase)[None]
            )
            value = torch.polar(magnitude, angle)
        elif self.bounded:
            magnitude = torch.exp(
                0.4 * torch.tanh(self.bounded_log_magnitude)
            )
            angle = math.pi * torch.tanh(self.bounded_phase)
            value = torch.polar(magnitude, angle)
        else:
            value = torch.complex(
                1.0 + self.unconstrained_real,
                self.unconstrained_imag,
            )
        value = value[:, :, self.band_index]
        return value.to(device=device, dtype=dtype).unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        lead_days: torch.Tensor,
    ) -> torch.Tensor:
        coeff = self.sht(x.float())
        latent = torch.einsum(
            "rc,bclm->brlm",
            self.encode_weight.to(coeff.dtype),
            coeff,
        )
        multiplier = self._lead_multiplier(
            lead_days,
            latent.dtype,
            latent.device,
        )
        evolved = latent[:, None] * multiplier[None]
        delta_coeff = torch.einsum(
            "cr,bnrlm->bnclm",
            self.decode_weight.to(evolved.dtype),
            evolved - latent[:, None],
        )
        b, n, c, ell, order = delta_coeff.shape
        delta = self.isht(delta_coeff.reshape(b * n * c, ell, order))
        return delta.reshape(b, n, c, x.shape[-2], x.shape[-1])


class SphereDynForecast(nn.Module):
    """Stable spherical-band dynamics plus a local weather residual branch."""

    def __init__(
        self,
        in_channels: int,
        out_vars: int,
        n_leads: int,
        img_size: tuple[int, int] = (121, 240),
        width: int = 128,
        local_layers: int = 6,
        spectral_rank: int = 48,
        spectral_lmax: int = 64,
        spectral_bands: int = 8,
        use_spectral: bool = True,
        use_local: bool = True,
        stable_spectral: bool = True,
        bounded_spectral: bool = False,
    ):
        super().__init__()
        if in_channels != out_vars:
            raise ValueError("SphereDyn requires in_channels == out_vars")
        if not use_spectral and not use_local:
            raise ValueError("SphereDyn needs at least one prediction branch")
        self.n_leads = int(n_leads)
        self.out_vars = int(out_vars)
        self.img_size = tuple(img_size)
        self.use_spectral = bool(use_spectral)
        self.use_local = bool(use_local)
        self.register_buffer(
            "lead_days",
            torch.tensor([1.0, 3.0, 5.0, 7.0, 10.0])[:n_leads],
            persistent=True,
        )
        if self.use_spectral:
            self.spectral = StableSphericalBandDynamics(
                out_vars,
                n_leads,
                img_size[0],
                img_size[1],
                rank=spectral_rank,
                lmax=spectral_lmax,
                mmax=spectral_lmax,
                n_bands=spectral_bands,
                stable=stable_spectral,
                bounded=bounded_spectral,
            )
            self.spectral_gate = nn.Parameter(torch.tensor(0.1))
        if self.use_local:
            # Add sin(latitude), cos(latitude), and normalized longitude.
            lat = torch.linspace(
                math.pi / 2,
                -math.pi / 2,
                img_size[0],
            ).view(1, 1, img_size[0], 1)
            lon = torch.linspace(
                -1.0,
                1.0,
                img_size[1],
            ).view(1, 1, 1, img_size[1])
            coords = torch.cat(
                [
                    torch.sin(lat).expand(1, 1, *img_size),
                    torch.cos(lat).expand(1, 1, *img_size),
                    lon.expand(1, 1, *img_size),
                ],
                dim=1,
            )
            self.register_buffer("coords", coords, persistent=True)
            self.local_lift = nn.Conv2d(in_channels + 3, width, 1)
            self.local_blocks = nn.Sequential(
                *[
                    SphericalDepthwiseBlock(width)
                    for _ in range(int(local_layers))
                ]
            )
            self.local_head = nn.Sequential(
                nn.GroupNorm(min(8, width), width),
                nn.GELU(),
                nn.Conv2d(width, n_leads * out_vars, 1),
            )
            nn.init.zeros_(self.local_head[-1].weight)
            nn.init.zeros_(self.local_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                f"SphereDyn expects {self.img_size}, got {tuple(x.shape[-2:])}"
            )
        prediction = x[:, None].expand(
            -1,
            self.n_leads,
            -1,
            -1,
            -1,
        )
        if self.use_spectral:
            prediction = prediction + torch.tanh(
                self.spectral_gate
            ) * self.spectral(x, self.lead_days)
        if self.use_local:
            coords = self.coords.expand(x.shape[0], -1, -1, -1)
            feature = self.local_lift(torch.cat([x, coords], dim=1))
            feature = self.local_blocks(feature)
            local = self.local_head(feature).view(
                x.shape[0],
                self.n_leads,
                self.out_vars,
                *self.img_size,
            )
            prediction = prediction + local
        return prediction


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
    if name in {
        "spheredyn_v9_multiscale",
        "spheredyn_v9_multiscale_null_control",
    }:
        from .spheredyn_v9 import SphereDynV9Forecast

        return SphereDynV9Forecast(
            in_channels,
            out_vars,
            n_leads,
            observation_width=192,
            observation_layers=6,
            observation_rank=80,
            observation_lmax=80,
            observation_bands=10,
            multiscale_mode=(
                "learned"
                if name == "spheredyn_v9_multiscale"
                else "null_control"
            ),
        )
    if name == "cirt":
        from .external_models import CirTForecast

        return CirTForecast(in_channels, out_vars, n_leads)
    if name == "climode":
        from .external_models import ClimODEForecast

        return ClimODEForecast(in_channels, out_vars, n_leads)
    raise ValueError(f"Unknown baseline model: {name}")
