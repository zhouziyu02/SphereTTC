from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_harmonics import InverseRealSHT, RealSHT

from .models import SphericalDepthwiseBlock, StableSphericalBandDynamics
from .spheredyn_v2 import (
    HarmonicClimateTendency,
    PeriodicSemiLagrangianTransport,
    seasonal_harmonics,
)


class SphericalHistoryDynamics(nn.Module):
    """Integrate finite-difference initial tendencies in spherical space.

    First and second backward differences are encoded jointly.  Every latent
    spherical band follows a bounded, damped response whose time integral
    saturates rather than extrapolating a noisy tendency linearly forever.
    """

    def __init__(
        self,
        channels: int,
        n_leads: int,
        nlat: int,
        nlon: int,
        rank: int = 96,
        lmax: int = 80,
        n_bands: int = 10,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_leads = int(n_leads)
        self.rank = int(rank)
        self.lmax = min(int(lmax), int(nlat))
        self.mmax = min(self.lmax, int(nlon) // 2 + 1)
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
        scale = 1.0 / math.sqrt(max(1, 2 * channels))
        self.encode_weight = nn.Parameter(
            scale * torch.randn(rank, 2 * channels)
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
        # softplus(log_timescale) + 1 initializes to roughly four days.
        self.log_timescale = nn.Parameter(
            torch.full((rank, n_bands), math.log(math.expm1(3.0)))
        )
        self.response_amplitude = nn.Parameter(
            torch.zeros(rank, n_bands)
        )
        self.phase = nn.Parameter(torch.zeros(rank, n_bands))

    def forward(
        self,
        first_difference: torch.Tensor,
        second_difference: torch.Tensor,
        lead_days: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat(
            [first_difference, second_difference],
            dim=1,
        )
        coefficients = self.sht(stacked.float())
        latent = torch.einsum(
            "rc,bclm->brlm",
            self.encode_weight.to(coefficients.dtype),
            coefficients,
        )
        lead = lead_days.to(
            dtype=torch.float32,
            device=latent.device,
        )
        timescale = 1.0 + F.softplus(self.log_timescale)
        integrated = timescale[None] * (
            1.0 - torch.exp(-lead[:, None, None] / timescale[None])
        )
        magnitude = integrated * (
            1.0 + 0.5 * torch.tanh(self.response_amplitude)[None]
        )
        angle = (
            math.pi
            * (lead / lead.max().clamp_min(1.0))[:, None, None]
            * torch.tanh(self.phase)[None]
        )
        response = torch.polar(magnitude, angle)
        response = response[:, :, self.band_index].unsqueeze(-1)
        evolved = latent[:, None] * response[None].to(latent.dtype)
        output_coefficients = torch.einsum(
            "cr,bnrlm->bnclm",
            self.decode_weight.to(evolved.dtype),
            evolved,
        )
        batch, leads, channels, ell, order = output_coefficients.shape
        output = self.isht(
            output_coefficients.reshape(
                batch * leads * channels,
                ell,
                order,
            )
        )
        return output.reshape(
            batch,
            leads,
            channels,
            first_difference.shape[-2],
            first_difference.shape[-1],
        )


class SphereDynV4Forecast(nn.Module):
    """History-conditioned spherical dynamics forecast.

    The model consumes three chronological daily states.  Its two new inputs
    are the first and second backward finite differences; the
    ``history_mode='current_only'`` control retains identical parameters while
    setting both differences to zero.
    """

    history_steps = 3
    requires_seasonal_context = True

    def __init__(
        self,
        in_channels: int,
        out_variables: int,
        n_leads: int,
        image_size: tuple[int, int] = (121, 240),
        width: int = 384,
        local_layers: int = 10,
        spectral_rank: int = 96,
        spectral_lmax: int = 80,
        spectral_bands: int = 10,
        history_mode: str = "finite_difference",
    ):
        super().__init__()
        if in_channels != out_variables:
            raise ValueError("SphereDyn-v4 requires equal input/output variables")
        if history_mode not in {"finite_difference", "current_only"}:
            raise ValueError(
                "history_mode must be 'finite_difference' or 'current_only'"
            )
        self.n_leads = int(n_leads)
        self.out_variables = int(out_variables)
        self.image_size = tuple(map(int, image_size))
        self.history_mode = str(history_mode)
        self.branch_count = 5
        lead_days = torch.tensor(
            [1.0, 3.0, 5.0, 7.0, 10.0][: self.n_leads]
        )
        self.register_buffer("lead_days", lead_days, persistent=True)

        lat = torch.linspace(
            math.pi / 2,
            -math.pi / 2,
            self.image_size[0],
        ).view(1, 1, self.image_size[0], 1)
        lon = torch.linspace(
            -1.0,
            1.0,
            self.image_size[1],
        ).view(1, 1, 1, self.image_size[1])
        coords = torch.cat(
            [
                torch.sin(lat).expand(1, 1, *self.image_size),
                torch.cos(lat).expand(1, 1, *self.image_size),
                lon.expand(1, 1, *self.image_size),
            ],
            dim=1,
        )
        self.register_buffer("coords", coords, persistent=True)
        self.local_lift = nn.Conv2d(
            3 * in_channels + 3 + 4,
            width,
            1,
        )
        self.local_blocks = nn.Sequential(
            *[SphericalDepthwiseBlock(width) for _ in range(local_layers)]
        )
        self.local_head = nn.Sequential(
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
            nn.Conv2d(width, n_leads * out_variables, 1),
        )
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

        self.state_spectral = StableSphericalBandDynamics(
            out_variables,
            n_leads,
            self.image_size[0],
            self.image_size[1],
            rank=spectral_rank,
            lmax=spectral_lmax,
            mmax=spectral_lmax,
            n_bands=spectral_bands,
            stable=False,
        )
        self.history_spectral = SphericalHistoryDynamics(
            out_variables,
            n_leads,
            self.image_size[0],
            self.image_size[1],
            rank=spectral_rank,
            lmax=spectral_lmax,
            n_bands=spectral_bands,
        )
        self.transport = PeriodicSemiLagrangianTransport(
            width,
            n_leads,
            out_variables,
            self.image_size,
        )
        self.climate_tendency = HarmonicClimateTendency(
            out_variables,
            self.image_size,
        )

        initial_values = [
            0.0,
            -2.944438979,
            -3.663561647,
            -1.945910149,
            -1.098612289,
        ]
        initial_gate = torch.tensor(initial_values, dtype=torch.float32)
        self.branch_logits = nn.Parameter(
            initial_gate.view(1, 1, self.branch_count).expand(
                n_leads,
                out_variables,
                self.branch_count,
            ).clone()
        )
        self.context_router = nn.Linear(
            width,
            n_leads * self.branch_count,
        )
        self.seasonal_router = nn.Linear(4, self.branch_count)
        nn.init.zeros_(self.context_router.weight)
        nn.init.zeros_(self.context_router.bias)
        nn.init.zeros_(self.seasonal_router.weight)
        nn.init.zeros_(self.seasonal_router.bias)

    def forward(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != self.history_steps:
            raise ValueError(
                "SphereDyn-v4 expects (batch, 3, variable, lat, lon)"
            )
        if tuple(history.shape[-2:]) != self.image_size:
            raise ValueError(
                f"SphereDyn-v4 expects {self.image_size}, "
                f"got {tuple(history.shape[-2:])}"
            )
        older, previous, current = history.unbind(dim=1)
        first_difference = current - previous
        second_difference = current - 2.0 * previous + older
        if self.history_mode == "current_only":
            first_difference = torch.zeros_like(first_difference)
            second_difference = torch.zeros_like(second_difference)
        if seasonal_context is None:
            seasonal_context = torch.zeros(
                (history.shape[0], 4),
                dtype=history.dtype,
                device=history.device,
            )
        seasonal = seasonal_context.to(
            dtype=history.dtype,
            device=history.device,
        )
        target_seasonal = seasonal_harmonics(seasonal, self.lead_days)
        coords = self.coords.expand(history.shape[0], -1, -1, -1)
        seasonal_map = seasonal[:, :, None, None].expand(
            -1,
            -1,
            *self.image_size,
        )
        feature = self.local_blocks(
            self.local_lift(
                torch.cat(
                    [
                        current,
                        first_difference,
                        second_difference,
                        coords,
                        seasonal_map,
                    ],
                    dim=1,
                )
            )
        )
        local_delta = self.local_head(feature).view(
            history.shape[0],
            self.n_leads,
            self.out_variables,
            *self.image_size,
        )
        state_delta = self.state_spectral(current, self.lead_days)
        transport_delta = self.transport(current, feature)
        history_delta = self.history_spectral(
            first_difference,
            second_difference,
            self.lead_days,
        )
        climate_delta = self.climate_tendency(
            seasonal,
            target_seasonal,
        )

        logits = self.branch_logits[None].expand(
            history.shape[0],
            -1,
            -1,
            -1,
        )
        context_offset = self.context_router(
            feature.mean(dim=(-2, -1))
        ).view(
            history.shape[0],
            self.n_leads,
            1,
            self.branch_count,
        )
        seasonal_offset = self.seasonal_router(
            target_seasonal
        ).unsqueeze(2)
        logits = (
            logits
            + 0.25 * torch.tanh(context_offset)
            + 0.25 * torch.tanh(seasonal_offset)
        )
        gates = 2.0 * torch.sigmoid(logits)
        correction = (
            gates[..., 0, None, None] * local_delta
            + gates[..., 1, None, None] * state_delta
            + gates[..., 2, None, None] * transport_delta
            + gates[..., 3, None, None] * history_delta
            + gates[..., 4, None, None] * climate_delta
        )
        return current[:, None] + correction
