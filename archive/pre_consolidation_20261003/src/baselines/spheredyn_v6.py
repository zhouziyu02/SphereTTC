from __future__ import annotations

import math

import torch
from torch import nn

from .models import SphericalDepthwiseBlock, StableSphericalBandDynamics
from .spheredyn_v2 import (
    HarmonicClimateTendency,
    PeriodicSemiLagrangianTransport,
    seasonal_harmonics,
)


class SphereDynV6Forecast(nn.Module):
    """A shared one-day spherical flow operator integrated to all horizons.

    Unlike the earlier direct multi-horizon models, one parameter-shared daily
    operator is recursively evaluated for ten days.  The ``frozen_tendency``
    mode has exactly the same parameters and operator calls, but evaluates
    every daily tendency at the initial state; it is the capacity- and
    compute-matched control for state-dependent recursive integration.
    """

    history_steps = 3
    requires_seasonal_context = True

    def __init__(
        self,
        in_channels: int,
        out_variables: int,
        n_leads: int,
        image_size: tuple[int, int] = (121, 240),
        width: int = 128,
        local_layers: int = 6,
        spectral_rank: int = 64,
        spectral_lmax: int = 64,
        spectral_bands: int = 8,
        integration_mode: str = "recursive",
    ):
        super().__init__()
        if in_channels != out_variables:
            raise ValueError("SphereDyn-v6 requires equal input/output variables")
        if integration_mode not in {"recursive", "frozen_tendency"}:
            raise ValueError(
                "integration_mode must be 'recursive' or 'frozen_tendency'"
            )
        available_steps = (1, 3, 5, 7, 10)
        if not 1 <= int(n_leads) <= len(available_steps):
            raise ValueError("SphereDyn-v6 supports one to five standard leads")
        self.n_leads = int(n_leads)
        self.out_variables = int(out_variables)
        self.image_size = tuple(map(int, image_size))
        self.integration_mode = str(integration_mode)
        self.branch_count = 4
        output_steps = torch.tensor(
            available_steps[: self.n_leads], dtype=torch.int64
        )
        self.register_buffer("output_steps", output_steps, persistent=True)
        self.register_buffer(
            "daily_lead", torch.tensor([1.0]), persistent=False
        )

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
        self.local_lift = nn.Conv2d(2 * in_channels + 3 + 4, width, 1)
        self.local_blocks = nn.Sequential(
            *[SphericalDepthwiseBlock(width) for _ in range(local_layers)]
        )
        self.local_head = nn.Sequential(
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
            nn.Conv2d(width, out_variables, 1),
        )
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

        self.spectral = StableSphericalBandDynamics(
            out_variables,
            1,
            self.image_size[0],
            self.image_size[1],
            rank=spectral_rank,
            lmax=spectral_lmax,
            mmax=spectral_lmax,
            n_bands=spectral_bands,
            stable=False,
        )
        self.transport = PeriodicSemiLagrangianTransport(
            width,
            1,
            out_variables,
            self.image_size,
        )
        self.climate_tendency = HarmonicClimateTendency(
            out_variables,
            self.image_size,
        )
        initial_gate = torch.tensor(
            [0.0, -2.944438979, -3.663561647, -1.098612289],
            dtype=torch.float32,
        )
        self.branch_logits = nn.Parameter(
            initial_gate[None].expand(out_variables, -1).clone()
        )

    def _daily_delta(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        current_seasonal: torch.Tensor,
        target_seasonal: torch.Tensor,
    ) -> torch.Tensor:
        coords = self.coords.expand(current.shape[0], -1, -1, -1)
        seasonal_map = current_seasonal[:, :, None, None].expand(
            -1, -1, *self.image_size
        )
        feature = self.local_blocks(
            self.local_lift(
                torch.cat(
                    [current, current - previous, coords, seasonal_map],
                    dim=1,
                )
            )
        )
        local_delta = self.local_head(feature)
        spectral_delta = self.spectral(current, self.daily_lead)[:, 0]
        transport_delta = self.transport(current, feature)[:, 0]
        climate_delta = self.climate_tendency(
            current_seasonal,
            target_seasonal[:, None],
        )[:, 0]
        gates = 2.0 * torch.sigmoid(self.branch_logits)
        return (
            gates[:, 0, None, None] * local_delta
            + gates[:, 1, None, None] * spectral_delta
            + gates[:, 2, None, None] * transport_delta
            + gates[:, 3, None, None] * climate_delta
        )

    def forward(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 5 or history.shape[1] != self.history_steps:
            raise ValueError(
                "SphereDyn-v6 expects (batch, 3, variable, lat, lon)"
            )
        if tuple(history.shape[-2:]) != self.image_size:
            raise ValueError(
                f"SphereDyn-v6 expects {self.image_size}, "
                f"got {tuple(history.shape[-2:])}"
            )
        if seasonal_context is None:
            seasonal_context = torch.zeros(
                (history.shape[0], 4),
                dtype=history.dtype,
                device=history.device,
            )
        initial_seasonal = seasonal_context.to(
            dtype=history.dtype,
            device=history.device,
        )
        maximum_step = int(self.output_steps[-1].item())
        target_contexts = seasonal_harmonics(
            initial_seasonal,
            torch.arange(
                1,
                maximum_step + 1,
                dtype=history.dtype,
                device=history.device,
            ),
        )
        _, previous, current = history.unbind(dim=1)
        fixed_previous = previous
        fixed_current = current
        outputs = []
        requested = {int(value) for value in self.output_steps.tolist()}
        for step in range(1, maximum_step + 1):
            current_context = (
                initial_seasonal if step == 1 else target_contexts[:, step - 2]
            )
            target_context = target_contexts[:, step - 1]
            if self.integration_mode == "recursive":
                delta = self._daily_delta(
                    previous,
                    current,
                    current_context,
                    target_context,
                )
                previous, current = current, current + delta
            else:
                delta = self._daily_delta(
                    fixed_previous,
                    fixed_current,
                    current_context,
                    target_context,
                )
                current = current + delta
            if step in requested:
                outputs.append(current)
        return torch.stack(outputs, dim=1)
