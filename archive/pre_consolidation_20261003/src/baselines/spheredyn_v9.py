from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .models import SphericalDepthwiseBlock
from .spheredyn_v7 import SphereDynV7Forecast


class SphereDynV9Forecast(SphereDynV7Forecast):
    """Dual-path dynamics with a direct multi-scale spherical residual.

    The v8 recursive and observation-time paths are retained exactly.  A third
    path represents the same initial state and its two finite differences at
    native, half and quarter resolution.  Spherical local blocks act at each
    scale before a zero-initialized residual head fuses them at the native
    grid.  The explicit null control executes the identical path and owns the
    same parameters, but suppresses only its final addition.
    """

    def __init__(
        self,
        in_channels: int,
        out_variables: int,
        n_leads: int,
        image_size: tuple[int, int] = (121, 240),
        multiscale_width: int = 160,
        multiscale_layers: int = 3,
        multiscale_mode: str = "learned",
        **dual_path_kwargs,
    ) -> None:
        if multiscale_mode not in {"learned", "null_control"}:
            raise ValueError(
                "multiscale_mode must be 'learned' or 'null_control'"
            )
        super().__init__(
            in_channels,
            out_variables,
            n_leads,
            image_size=image_size,
            **dual_path_kwargs,
        )
        self.multiscale_mode = multiscale_mode
        context_channels = 3 * in_channels + 3 + 4
        self.multiscale_lift = nn.Conv2d(
            context_channels, multiscale_width, 1
        )
        self.multiscale_fine = nn.Sequential(
            *[
                SphericalDepthwiseBlock(multiscale_width)
                for _ in range(multiscale_layers)
            ]
        )
        self.multiscale_mid = nn.Sequential(
            *[
                SphericalDepthwiseBlock(multiscale_width)
                for _ in range(multiscale_layers)
            ]
        )
        self.multiscale_coarse = nn.Sequential(
            *[
                SphericalDepthwiseBlock(multiscale_width)
                for _ in range(multiscale_layers)
            ]
        )
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(3 * multiscale_width, multiscale_width, 1),
            nn.GELU(),
            SphericalDepthwiseBlock(multiscale_width),
        )
        self.multiscale_head = nn.Sequential(
            nn.GroupNorm(min(8, multiscale_width), multiscale_width),
            nn.GELU(),
            nn.Conv2d(multiscale_width, n_leads * out_variables, 1),
        )
        nn.init.zeros_(self.multiscale_head[-1].weight)
        nn.init.zeros_(self.multiscale_head[-1].bias)
        self.multiscale_logits = nn.Parameter(
            torch.full((n_leads, out_variables), -1.945910149)
        )

    def forward(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dual_path = super().forward(history, seasonal_context)
        older, previous, current = history.unbind(dim=1)
        first_difference = current - previous
        second_difference = current - 2.0 * previous + older
        if seasonal_context is None:
            seasonal_context = torch.zeros(
                (history.shape[0], 4),
                dtype=history.dtype,
                device=history.device,
            )
        seasonal = seasonal_context.to(
            dtype=history.dtype, device=history.device
        )
        seasonal_map = seasonal[:, :, None, None].expand(
            -1, -1, *self.image_size
        )
        context = torch.cat(
            [
                current,
                first_difference,
                second_difference,
                self.coords.expand(history.shape[0], -1, -1, -1),
                seasonal_map,
            ],
            dim=1,
        )
        lifted = self.multiscale_lift(context)
        fine = self.multiscale_fine(lifted)
        mid = self.multiscale_mid(F.avg_pool2d(lifted, 2, 2))
        coarse = self.multiscale_coarse(F.avg_pool2d(lifted, 4, 4))
        mid = F.interpolate(
            mid, size=self.image_size, mode="bilinear", align_corners=False
        )
        coarse = F.interpolate(
            coarse, size=self.image_size, mode="bilinear", align_corners=False
        )
        fused = self.multiscale_fusion(torch.cat([fine, mid, coarse], dim=1))
        residual = self.multiscale_head(fused).view(
            history.shape[0],
            self.n_leads,
            self.out_variables,
            *self.image_size,
        )
        residual = (
            2.0
            * torch.sigmoid(self.multiscale_logits)[..., None, None]
            * residual
        )
        if self.multiscale_mode == "null_control":
            residual = 0.0 * residual
        return dual_path + residual
