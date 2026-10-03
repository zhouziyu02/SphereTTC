from __future__ import annotations

import torch
from torch import nn

from .models import SphericalDepthwiseBlock
from .spheredyn_v2 import seasonal_harmonics
from .spheredyn_v4 import SphericalHistoryDynamics
from .spheredyn_v6 import SphereDynV6Forecast


class SphereDynV7Forecast(SphereDynV6Forecast):
    """Recursive flow plus a direct observation-time residual decoder.

    The recursive v6 path supplies a parameter-shared daily semigroup.  A
    compact second path decodes the initial state and its two backward finite
    differences directly at the five requested observation times.  Its heads
    are zero initialized, so a v6 checkpoint is an exact v7 no-op warm start.
    ``null_control`` executes the same operations and retains the same
    parameters but suppresses the direct residual at the final addition.
    """

    def __init__(
        self,
        in_channels: int,
        out_variables: int,
        n_leads: int,
        image_size: tuple[int, int] = (121, 240),
        observation_width: int = 96,
        observation_layers: int = 4,
        observation_rank: int = 48,
        observation_lmax: int = 64,
        observation_bands: int = 8,
        observation_mode: str = "learned",
        **recursive_kwargs,
    ) -> None:
        if observation_mode not in {"learned", "null_control"}:
            raise ValueError("observation_mode must be 'learned' or 'null_control'")
        super().__init__(
            in_channels,
            out_variables,
            n_leads,
            image_size=image_size,
            **recursive_kwargs,
        )
        self.observation_mode = observation_mode
        self.observation_lift = nn.Conv2d(
            3 * in_channels + 3 + 4,
            observation_width,
            1,
        )
        self.observation_blocks = nn.Sequential(
            *[
                SphericalDepthwiseBlock(observation_width)
                for _ in range(observation_layers)
            ]
        )
        self.observation_head = nn.Sequential(
            nn.GroupNorm(min(8, observation_width), observation_width),
            nn.GELU(),
            nn.Conv2d(observation_width, n_leads * out_variables, 1),
        )
        nn.init.zeros_(self.observation_head[-1].weight)
        nn.init.zeros_(self.observation_head[-1].bias)
        self.observation_history = SphericalHistoryDynamics(
            out_variables,
            n_leads,
            image_size[0],
            image_size[1],
            rank=observation_rank,
            lmax=observation_lmax,
            n_bands=observation_bands,
        )
        nn.init.zeros_(self.observation_history.decode_weight)
        initial_logits = torch.tensor([0.0, -1.945910149], dtype=torch.float32)
        self.observation_branch_logits = nn.Parameter(
            initial_logits.view(1, 1, 2).expand(n_leads, out_variables, -1).clone()
        )

    def forward(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recursive = super().forward(history, seasonal_context)
        older, previous, current = history.unbind(dim=1)
        first_difference = current - previous
        second_difference = current - 2.0 * previous + older
        if seasonal_context is None:
            seasonal_context = torch.zeros(
                (history.shape[0], 4),
                dtype=history.dtype,
                device=history.device,
            )
        seasonal = seasonal_context.to(dtype=history.dtype, device=history.device)
        # Materialize target harmonics as an explicit time-conditioning check;
        # the direct local decoder uses the initialization phase and its
        # separate output channels identify the observation times.
        _ = seasonal_harmonics(seasonal, self.output_steps.to(history.dtype))
        seasonal_map = seasonal[:, :, None, None].expand(
            -1, -1, *self.image_size
        )
        feature = self.observation_blocks(
            self.observation_lift(
                torch.cat(
                    [
                        current,
                        first_difference,
                        second_difference,
                        self.coords.expand(history.shape[0], -1, -1, -1),
                        seasonal_map,
                    ],
                    dim=1,
                )
            )
        )
        local = self.observation_head(feature).view(
            history.shape[0],
            self.n_leads,
            self.out_variables,
            *self.image_size,
        )
        observation_days = self.output_steps.to(
            dtype=history.dtype, device=history.device
        )
        history_delta = self.observation_history(
            first_difference,
            second_difference,
            observation_days,
        )
        gates = 2.0 * torch.sigmoid(self.observation_branch_logits)
        direct_residual = (
            gates[..., 0, None, None] * local
            + gates[..., 1, None, None] * history_delta
        )
        if self.observation_mode == "null_control":
            direct_residual = 0.0 * direct_residual
        return recursive + direct_residual
