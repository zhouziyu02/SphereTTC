from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import (
    SphericalDepthwiseBlock,
    StableSphericalBandDynamics,
)


def seasonal_harmonics(
    init_times: torch.Tensor,
    lead_days: torch.Tensor,
) -> torch.Tensor:
    """Rotate initialization-day harmonics to every target lead.

    ``init_times`` contains [sin(theta), cos(theta), sin(2 theta),
    cos(2 theta)]. The returned tensor is shaped (batch, lead, 4).
    """
    if init_times.ndim != 2 or init_times.shape[-1] != 4:
        raise ValueError("seasonal context must have shape (batch, 4)")
    phase = (
        2.0
        * math.pi
        * lead_days.to(device=init_times.device, dtype=init_times.dtype)
        / 365.2425
    )
    sin1, cos1, sin2, cos2 = init_times.unbind(dim=-1)
    target_sin1 = (
        sin1[:, None] * torch.cos(phase)[None]
        + cos1[:, None] * torch.sin(phase)[None]
    )
    target_cos1 = (
        cos1[:, None] * torch.cos(phase)[None]
        - sin1[:, None] * torch.sin(phase)[None]
    )
    phase2 = 2.0 * phase
    target_sin2 = (
        sin2[:, None] * torch.cos(phase2)[None]
        + cos2[:, None] * torch.sin(phase2)[None]
    )
    target_cos2 = (
        cos2[:, None] * torch.cos(phase2)[None]
        - sin2[:, None] * torch.sin(phase2)[None]
    )
    return torch.stack(
        [target_sin1, target_cos1, target_sin2, target_cos2],
        dim=-1,
    )


class PeriodicSemiLagrangianTransport(nn.Module):
    """Bounded backward transport with periodic longitude.

    The canonical variable order has five ten-level pressure groups followed by
    four surface variables. A single grid-sample call per lead transports all
    six groups while allowing a different learned flow for every group.
    """

    def __init__(
        self,
        feature_channels: int,
        n_leads: int,
        n_variables: int,
        image_size: tuple[int, int],
    ):
        super().__init__()
        if n_variables != 54:
            raise ValueError(
                "the transport grouping currently requires the canonical "
                "54-variable protocol"
            )
        self.n_leads = int(n_leads)
        self.n_variables = int(n_variables)
        self.height, self.width = map(int, image_size)
        self.group_count = 6
        self.group_width = 10
        self.flow_head = nn.Conv2d(
            feature_channels,
            self.n_leads * self.group_count * 2,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        maximum = torch.tensor(
            [4.0, 8.0, 12.0, 16.0, 20.0][: self.n_leads],
            dtype=torch.float32,
        )
        self.register_buffer("maximum_displacement", maximum, persistent=True)
        self.longitude_pad = int(math.ceil(float(maximum.max()))) + 1

        y = torch.arange(self.height, dtype=torch.float32).view(
            1, 1, self.height, 1
        )
        x = torch.arange(self.width, dtype=torch.float32).view(
            1, 1, 1, self.width
        )
        self.register_buffer("base_y", y, persistent=False)
        self.register_buffer("base_x", x, persistent=False)

    def _group(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        pressure = x[:, :50].reshape(
            batch,
            5,
            self.group_width,
            height,
            width,
        )
        surface = F.pad(
            x[:, 50:].unsqueeze(1),
            (0, 0, 0, 0, 0, self.group_width - 4),
        )
        return torch.cat([pressure, surface], dim=1)

    @staticmethod
    def _ungroup(x: torch.Tensor) -> torch.Tensor:
        pressure = x[:, :5].flatten(1, 2)
        surface = x[:, 5, :4]
        return torch.cat([pressure, surface], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        batch = x.shape[0]
        flows = self.flow_head(feature).view(
            batch,
            self.n_leads,
            self.group_count,
            2,
            self.height,
            self.width,
        )
        grouped = self._group(x)
        pad = self.longitude_pad
        periodic = torch.cat(
            [grouped[..., -pad:], grouped, grouped[..., :pad]],
            dim=-1,
        )
        padded_width = self.width + 2 * pad
        source = periodic.flatten(0, 1)
        outputs = []
        for lead_index in range(self.n_leads):
            maximum = self.maximum_displacement[lead_index]
            flow = torch.tanh(flows[:, lead_index]) * maximum
            delta_x = flow[:, :, 0]
            delta_y = flow[:, :, 1]
            sample_x = (
                torch.remainder(self.base_x - delta_x, self.width)
                + float(pad)
            )
            sample_y = (self.base_y - delta_y).clamp(
                0.0,
                float(self.height - 1),
            )
            normalized_x = (
                2.0 * sample_x / float(padded_width - 1) - 1.0
            )
            normalized_y = (
                2.0 * sample_y / float(self.height - 1) - 1.0
            )
            grid = torch.stack(
                [normalized_x, normalized_y],
                dim=-1,
            ).flatten(0, 1)
            transported = F.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(
                batch,
                self.group_count,
                self.group_width,
                self.height,
                self.width,
            )
            outputs.append(
                self._ungroup(transported - grouped)
            )
        return torch.stack(outputs, dim=1)


class HarmonicClimateTendency(nn.Module):
    """Low-rank learned day-of-year climatological tendency.

    A climatological field represented by annual and semi-annual harmonics has
    an exact lead tendency that is linear in the difference between target-day
    and initialization-day harmonic features. Learning the spatial bases on a
    coarse spherical grid keeps this branch compact and avoids storing a large
    day-of-year climatology inside every checkpoint.
    """

    def __init__(
        self,
        n_variables: int,
        image_size: tuple[int, int],
        basis_size: tuple[int, int] = (16, 30),
    ):
        super().__init__()
        self.n_variables = int(n_variables)
        self.image_size = tuple(map(int, image_size))
        self.basis = nn.Parameter(
            torch.zeros(4, self.n_variables, *basis_size)
        )

    def forward(
        self,
        initial_context: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        if initial_context.ndim != 2 or initial_context.shape[-1] != 4:
            raise ValueError("initial_context must have shape (batch, 4)")
        if target_context.ndim != 3 or target_context.shape[-1] != 4:
            raise ValueError(
                "target_context must have shape (batch, lead, 4)"
            )
        delta_context = target_context - initial_context[:, None]
        coarse = torch.einsum(
            "blh,hvxy->blvxy",
            delta_context,
            self.basis,
        )
        batch, leads, variables, height, width = coarse.shape
        full = F.interpolate(
            coarse.reshape(batch * leads, variables, height, width),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return full.reshape(
            batch,
            leads,
            variables,
            *self.image_size,
        )


class SphereDynV2Forecast(nn.Module):
    """Lead-aware spherical dynamics with optional transport and seasonality."""

    def __init__(
        self,
        in_channels: int,
        out_variables: int,
        n_leads: int,
        image_size: tuple[int, int] = (121, 240),
        width: int = 128,
        local_layers: int = 6,
        spectral_rank: int = 48,
        spectral_lmax: int = 64,
        spectral_bands: int = 8,
        use_router: bool = True,
        use_transport: bool = False,
        use_seasonal: bool = False,
        use_climate_tendency: bool = False,
        climate_tendency_mode: str = "seasonal",
    ):
        super().__init__()
        if in_channels != out_variables:
            raise ValueError("SphereDyn-v2 requires equal input/output variables")
        self.n_leads = int(n_leads)
        self.out_variables = int(out_variables)
        self.image_size = tuple(image_size)
        self.use_router = bool(use_router)
        self.use_transport = bool(use_transport)
        self.use_climate_tendency = bool(use_climate_tendency)
        if climate_tendency_mode not in {"seasonal", "lead_only"}:
            raise ValueError(
                "climate_tendency_mode must be 'seasonal' or 'lead_only'"
            )
        self.climate_tendency_mode = str(climate_tendency_mode)
        self.use_seasonal = bool(
            use_seasonal or self.use_climate_tendency
        )
        self.branch_count = 4 if self.use_climate_tendency else 3
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
        seasonal_channels = 4 if self.use_seasonal else 0
        self.local_lift = nn.Conv2d(
            in_channels + 3 + seasonal_channels,
            width,
            1,
        )
        self.local_blocks = nn.Sequential(
            *[
                SphericalDepthwiseBlock(width)
                for _ in range(int(local_layers))
            ]
        )
        self.local_head = nn.Sequential(
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
            nn.Conv2d(width, n_leads * out_variables, 1),
        )
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

        self.spectral = StableSphericalBandDynamics(
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
        if self.use_transport:
            self.transport = PeriodicSemiLagrangianTransport(
                width,
                n_leads,
                out_variables,
                self.image_size,
            )
        if self.use_climate_tendency:
            self.climate_tendency = HarmonicClimateTendency(
                out_variables,
                self.image_size,
            )

        # Two times sigmoid(logit) initializes local/spectral/transport/
        # climate gates to 1.0/0.1/0.05/0.5. The climate basis itself starts
        # at zero, so SphereDyn-v3 is initially functionally identical to v2.
        initial_values = [0.0, -2.944438979, -3.663561647]
        if self.use_climate_tendency:
            initial_values.append(-1.098612289)
        initial_gate = torch.tensor(initial_values, dtype=torch.float32)
        self.branch_logits = nn.Parameter(
            initial_gate.view(1, 1, self.branch_count).expand(
                n_leads,
                out_variables,
                self.branch_count,
            ).clone()
        )
        if self.use_router:
            self.context_router = nn.Linear(
                width,
                n_leads * self.branch_count,
            )
            nn.init.zeros_(self.context_router.weight)
            nn.init.zeros_(self.context_router.bias)
            if self.use_seasonal:
                self.seasonal_router = nn.Linear(
                    4,
                    self.branch_count,
                )
                nn.init.zeros_(self.seasonal_router.weight)
                nn.init.zeros_(self.seasonal_router.bias)

    @property
    def requires_seasonal_context(self) -> bool:
        return self.use_seasonal

    def _seasonal(
        self,
        x: torch.Tensor,
        seasonal_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seasonal_context is None:
            seasonal_context = torch.zeros(
                (x.shape[0], 4),
                dtype=x.dtype,
                device=x.device,
            )
        seasonal_context = seasonal_context.to(
            device=x.device,
            dtype=x.dtype,
        )
        target_context = seasonal_harmonics(
            seasonal_context,
            self.lead_days,
        )
        return seasonal_context, target_context

    def forward(
        self,
        x: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"SphereDyn-v2 expects {self.image_size}, "
                f"got {tuple(x.shape[-2:])}"
            )
        seasonal, target_seasonal = self._seasonal(
            x,
            seasonal_context,
        )
        coords = self.coords.expand(x.shape[0], -1, -1, -1)
        local_inputs = [x, coords]
        if self.use_seasonal:
            local_inputs.append(
                seasonal[:, :, None, None].expand(
                    -1,
                    -1,
                    *self.image_size,
                )
            )
        feature = self.local_blocks(
            self.local_lift(torch.cat(local_inputs, dim=1))
        )
        local_delta = self.local_head(feature).view(
            x.shape[0],
            self.n_leads,
            self.out_variables,
            *self.image_size,
        )
        spectral_delta = self.spectral(x, self.lead_days)
        if self.use_transport:
            transport_delta = self.transport(x, feature)
        else:
            transport_delta = torch.zeros_like(local_delta)
        if self.use_climate_tendency:
            climate_initial = seasonal
            climate_target = target_seasonal
            if self.climate_tendency_mode == "lead_only":
                # Exact-capacity control: retain the same fourth branch,
                # router dimensions and spatial harmonic bases, but remove
                # initialization-date dependence from the tendency field.
                climate_initial = torch.zeros_like(seasonal)
                climate_initial[:, 1] = 1.0
                climate_initial[:, 3] = 1.0
                climate_target = seasonal_harmonics(
                    climate_initial,
                    self.lead_days,
                )
            climate_delta = self.climate_tendency(
                climate_initial,
                climate_target,
            )
        else:
            climate_delta = torch.zeros_like(local_delta)

        logits = self.branch_logits[None].expand(
            x.shape[0],
            -1,
            -1,
            -1,
        )
        if self.use_router:
            context = self.context_router(
                feature.mean(dim=(-2, -1))
            ).view(
                x.shape[0],
                self.n_leads,
                1,
                self.branch_count,
            )
            logits = logits + 0.25 * torch.tanh(context)
            if self.use_seasonal:
                seasonal_offset = self.seasonal_router(
                    target_seasonal
                ).unsqueeze(2)
                logits = logits + 0.25 * torch.tanh(seasonal_offset)
        gates = 2.0 * torch.sigmoid(logits)
        if not self.use_transport:
            gates = gates.clone()
            gates[..., 2] = 0.0
        correction = (
            gates[..., 0, None, None] * local_delta
            + gates[..., 1, None, None] * spectral_delta
            + gates[..., 2, None, None] * transport_delta
        )
        if self.use_climate_tendency:
            correction = (
                correction
                + gates[..., 3, None, None] * climate_delta
            )
        persistence = x[:, None].expand(
            -1,
            self.n_leads,
            -1,
            -1,
            -1,
        )
        return persistence + correction
