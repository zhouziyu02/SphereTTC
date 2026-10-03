"""SphereDyn: a spherical-dynamics backbone for medium-range weather forecasting.

This single file contains the complete SphereDyn model used in the paper
(``SphereDyn-v9 multiscale``, seed 44, 3,966,080 parameters).  The frozen
checkpoint is

    artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/
        spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt

and loads into :class:`SphereDyn` with ``strict=True`` (see ``build_model`` in
``src/baselines/models.py`` and ``tests/test_spheredyn_checkpoint.py``).

Tensors (B = batch, V = 54 variables, H x W = 121 x 240 equiangular 1.5 deg grid)
----------------------------------------------------------------------------------
    history           (B, 3, V, H, W)  normalized daily-mean states at days t-2, t-1, t
    seasonal_context  (B, 4)           [sin a, cos a, sin 2a, cos 2a], a = 2*pi*day_of_year/366
    output            (B, L, V, H, W)  normalized daily-mean forecasts at L = 5 observation
                                       times: +1, +3, +5, +7 and +10 days (24 ... 240 h)

Architecture: three additive paths
----------------------------------
    y(tau) = RecursiveFlow(tau) + ObservationTimePath(tau) + MultiResolutionResidual(tau)

1. Recursive spherical flow  (``_recursive_flow``; paper: "parameter-shared daily
   flow operator").  One daily operator Phi is applied ten times,
       x_{d+1} = x_d + Phi(x_{d-1}, x_d, season_d -> season_{d+1}),   d = 0 ... 9,
   and x_1, x_3, x_5, x_7, x_10 are returned.  Phi (``_daily_delta``) is a gated
   sum of four branches with per-variable gates 2*sigmoid(branch_logits):
     a. spherical local branch   lift(1x1) -> 6 x SphericalLocalBlock(128) -> head
                                 on [x_d, x_d - x_{d-1}, coords, season]
     b. spectral branch          SphericalSpectralDynamics: truncated spherical
                                 harmonics (lmax = mmax = 64), rank-64 latent, 8 l-bands
     c. transport branch         SemiLagrangianTransport: a displacement field per
                                 variable group, predicted from branch a's features,
                                 bilinear backward advection with periodic longitude
     d. climate branch           HarmonicClimateTendency: annual + semi-annual
                                 harmonic tendency on a coarse 16 x 30 basis
2. Direct observation-time path  (``_observation_time_residual``; paper: "direct
   observation-time pathway").  Decodes the state and its finite differences
   D1 = x_t - x_{t-1}, D2 = x_t - 2 x_{t-1} + x_{t-2} directly at the five
   observation times, through
     a. a local decoder          lift -> 6 x SphericalLocalBlock(192) -> head (L*V channels)
     b. SphericalHistoryResponse truncated spherical harmonics (lmax = 80), rank 80,
                                 10 l-bands, damped/saturating response per lead
   combined with gates 2*sigmoid(observation_branch_logits) per (lead, variable).
3. Multi-resolution spherical residual  (``_multiscale_residual``; paper: "multi-
   resolution spherical residual pathway").  The same context at native, 1/2 and
   1/4 resolution -> 3 x SphericalLocalBlock(160) per scale -> upsample -> fuse ->
   head, gated by 2*sigmoid(multiscale_logits) per (lead, variable).

All output heads of paths 2 and 3 are zero-initialized, so each newer path starts
as an exact no-op on top of the previous model (warm starts v6 -> v7/v8 -> v9).

Ablation switches (defaults = paper model): ``integration_mode="frozen_tendency"``
evaluates every daily tendency at the initial state; ``observation_mode`` /
``multiscale_mode = "null_control"`` keep the parameters and compute of a path but
suppress its final addition.

Provenance
----------
Behaviour-identical merge of src/baselines/models.py (SphericalDepthwiseBlock,
StableSphericalBandDynamics), spheredyn_v2.py (seasonal_harmonics,
PeriodicSemiLagrangianTransport, HarmonicClimateTendency), spheredyn_v4.py
(SphericalHistoryDynamics), spheredyn_v6/v7/v9.py (SphereDynV6/V7/V9Forecast),
archived byte-identically in archive/pre_consolidation_20261003/.  Classes were
renamed for readability; parameter and buffer names are unchanged, the random
initialization order is unchanged, and tests/test_consolidation_equivalence.py
checks bitwise-identical outputs against the archived originals.

    old name                          -> name in this file
    SphericalDepthwiseBlock           -> SphericalLocalBlock
    StableSphericalBandDynamics       -> SphericalSpectralDynamics (only the
                                         unconstrained daily mode used by SphereDyn)
    PeriodicSemiLagrangianTransport   -> SemiLagrangianTransport
    SphericalHistoryDynamics          -> SphericalHistoryResponse
    SphereDynV9Forecast (+V7, V6)     -> SphereDyn

Run ``python -m src.spheredyn`` to print the module tree and parameter counts.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_harmonics import InverseRealSHT, RealSHT


# =============================================================================
# 1. Spherical building blocks
# =============================================================================


def _spherical_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Periodically pad longitude and replicate the polar latitude boundary."""
    if pad <= 0:
        return x
    x = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    return F.pad(x, (0, 0, pad, pad), mode="replicate")


class SphericalLocalBlock(nn.Module):
    """Spherical local residual block (no false periodic latitude boundary).

    x -> GroupNorm -> spherical pad -> depthwise conv (5x5) -> 1x1 expand to
    (value, gate) -> GELU(value) * sigmoid(gate) -> 1x1 project (zero init) -> + x
    """

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


class SphericalSpectralDynamics(nn.Module):
    """Low-rank dynamics in a truncated spherical-harmonic basis (branch 1b).

    x (B, C, H, W) --SHT--> a[c, l, m]          (l < lmax, m < mmax)
    z = E a                                     E: (rank, C) encoder
    z' = (1 + A + iB)[r, band(l)] * z           one complex multiplier per latent
                                                mode r and l-band (n_bands bands)
    dx = ISHT(D (z' - z))                       D: (C, rank) decoder

    Returned with a lead axis: (B, n_leads, C, H, W); SphereDyn uses n_leads = 1
    (one day).  This is the unconstrained ("stable=False") mode of the original
    StableSphericalBandDynamics, the only mode used by SphereDyn.
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
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_leads = int(n_leads)
        self.rank = int(rank)
        self.lmax = min(int(lmax), int(nlat))
        self.mmax = min(int(mmax), int(nlon) // 2 + 1, self.lmax)
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


class SemiLagrangianTransport(nn.Module):
    """Bounded backward (semi-Lagrangian) transport with periodic longitude (branch 1c).

    The canonical variable order has five ten-level pressure groups (z, q, t, u,
    v) followed by four surface variables, i.e. six groups.  A 3x3 conv flow head
    predicts a displacement (dx, dy) per group and lead from the local-branch
    features; displacements are bounded by tanh * {4, 8, 12, 16, 20} grid cells;
    a single grid-sample call per lead transports all six groups.  The branch
    returns the transport increment  x(departure point) - x.
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
    """Low-rank learned day-of-year climatological tendency (branch 1d).

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


class SphericalHistoryResponse(nn.Module):
    """Integrate finite-difference initial tendencies in spherical space (path 2b).

    First and second backward differences are encoded jointly.  Every latent
    spherical band follows a bounded, damped response whose time integral
    saturates rather than extrapolating a noisy tendency linearly forever:

        z = E SHT([D1, D2])                          E: (rank, 2C)
        R(tau) = T (1 - exp(-tau / T)) (1 + 0.5 tanh A) exp(i pi tau/tau_max tanh P)
        y(tau) = ISHT(D (R(tau)[r, band(l)] * z))    D: (C, rank)

    with a learnable timescale T = 1 + softplus(.) per latent mode and l-band.
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


# =============================================================================
# 2. SphereDyn
# =============================================================================


class SphereDyn(nn.Module):
    """SphereDyn = recursive spherical flow + observation-time path + multi-resolution residual.

    ``SphereDyn(54, 54, 5)`` builds exactly the paper model (all defaults are the
    frozen configuration).  See the module docstring for the full data flow.
    """

    history_steps = 3
    requires_seasonal_context = True

    def __init__(
        self,
        in_channels: int,
        out_variables: int,
        n_leads: int,
        image_size: tuple[int, int] = (121, 240),
        # path 1: recursive daily flow operator
        width: int = 128,
        local_layers: int = 6,
        spectral_rank: int = 64,
        spectral_lmax: int = 64,
        spectral_bands: int = 8,
        integration_mode: str = "recursive",
        # path 2: direct observation-time path
        observation_width: int = 192,
        observation_layers: int = 6,
        observation_rank: int = 80,
        observation_lmax: int = 80,
        observation_bands: int = 10,
        observation_mode: str = "learned",
        # path 3: multi-resolution spherical residual
        multiscale_width: int = 160,
        multiscale_layers: int = 3,
        multiscale_mode: str = "learned",
    ) -> None:
        if multiscale_mode not in {"learned", "null_control"}:
            raise ValueError(
                "multiscale_mode must be 'learned' or 'null_control'"
            )
        if observation_mode not in {"learned", "null_control"}:
            raise ValueError("observation_mode must be 'learned' or 'null_control'")
        super().__init__()
        if in_channels != out_variables:
            raise ValueError("SphereDyn requires equal input/output variables")
        if integration_mode not in {"recursive", "frozen_tendency"}:
            raise ValueError(
                "integration_mode must be 'recursive' or 'frozen_tendency'"
            )
        available_steps = (1, 3, 5, 7, 10)
        if not 1 <= int(n_leads) <= len(available_steps):
            raise ValueError("SphereDyn supports one to five standard leads")

        # ---- path 1: recursive spherical flow --------------------------------
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
        # 1a. spherical local branch on [x_d, x_d - x_{d-1}, coords(3), season(4)]
        self.local_lift = nn.Conv2d(2 * in_channels + 3 + 4, width, 1)
        self.local_blocks = nn.Sequential(
            *[SphericalLocalBlock(width) for _ in range(local_layers)]
        )
        self.local_head = nn.Sequential(
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
            nn.Conv2d(width, out_variables, 1),
        )
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)
        # 1b. truncated spherical-harmonic spectral branch (one-day operator)
        self.spectral = SphericalSpectralDynamics(
            out_variables,
            1,
            self.image_size[0],
            self.image_size[1],
            rank=spectral_rank,
            lmax=spectral_lmax,
            mmax=spectral_lmax,
            n_bands=spectral_bands,
        )
        # 1c. semi-Lagrangian transport driven by the local-branch features
        self.transport = SemiLagrangianTransport(
            width,
            1,
            out_variables,
            self.image_size,
        )
        # 1d. harmonic climate tendency
        self.climate_tendency = HarmonicClimateTendency(
            out_variables,
            self.image_size,
        )
        # per-variable branch gates g = 2 * sigmoid(logits): [local, spectral, transport, climate]
        initial_gate = torch.tensor(
            [0.0, -2.944438979, -3.663561647, -1.098612289],
            dtype=torch.float32,
        )
        self.branch_logits = nn.Parameter(
            initial_gate[None].expand(out_variables, -1).clone()
        )

        # ---- path 2: direct observation-time path ----------------------------
        self.observation_mode = observation_mode
        self.observation_lift = nn.Conv2d(
            3 * in_channels + 3 + 4,
            observation_width,
            1,
        )
        self.observation_blocks = nn.Sequential(
            *[
                SphericalLocalBlock(observation_width)
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
        self.observation_history = SphericalHistoryResponse(
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

        # ---- path 3: multi-resolution spherical residual ---------------------
        self.multiscale_mode = multiscale_mode
        context_channels = 3 * in_channels + 3 + 4
        self.multiscale_lift = nn.Conv2d(
            context_channels, multiscale_width, 1
        )
        self.multiscale_fine = nn.Sequential(
            *[
                SphericalLocalBlock(multiscale_width)
                for _ in range(multiscale_layers)
            ]
        )
        self.multiscale_mid = nn.Sequential(
            *[
                SphericalLocalBlock(multiscale_width)
                for _ in range(multiscale_layers)
            ]
        )
        self.multiscale_coarse = nn.Sequential(
            *[
                SphericalLocalBlock(multiscale_width)
                for _ in range(multiscale_layers)
            ]
        )
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(3 * multiscale_width, multiscale_width, 1),
            nn.GELU(),
            SphericalLocalBlock(multiscale_width),
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

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recursive = self._recursive_flow(history, seasonal_context)
        dual_path = recursive + self._observation_time_residual(
            history, seasonal_context
        )
        return dual_path + self._multiscale_residual(history, seasonal_context)

    # ---- path 1 -------------------------------------------------------------
    def _daily_delta(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        current_seasonal: torch.Tensor,
        target_seasonal: torch.Tensor,
    ) -> torch.Tensor:
        """One application of the daily operator Phi: returns x_{d+1} - x_d."""
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

    def _recursive_flow(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate Phi for ten days and read out days 1, 3, 5, 7, 10."""
        if history.ndim != 5 or history.shape[1] != self.history_steps:
            raise ValueError(
                "SphereDyn expects (batch, 3, variable, lat, lon)"
            )
        if tuple(history.shape[-2:]) != self.image_size:
            raise ValueError(
                f"SphereDyn expects {self.image_size}, "
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

    # ---- path 2 -------------------------------------------------------------
    def _observation_time_residual(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode [x_t, D1, D2] directly at the five observation times."""
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
        return direct_residual

    # ---- path 3 -------------------------------------------------------------
    def _multiscale_residual(
        self,
        history: torch.Tensor,
        seasonal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Native / half / quarter-resolution spherical residual, fused at native grid."""
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
        return residual


def parameter_breakdown(model: SphereDyn) -> list[tuple[str, int]]:
    """Parameter counts grouped by path/branch (for figures and tables)."""
    groups = {
        "1a local branch (lift + 6 blocks + head)": ("local_lift", "local_blocks", "local_head"),
        "1b spectral branch (SphericalSpectralDynamics)": ("spectral",),
        "1c semi-Lagrangian transport (flow head)": ("transport",),
        "1d harmonic climate tendency": ("climate_tendency",),
        "1  branch gates": ("branch_logits",),
        "2a observation-time local decoder": ("observation_lift", "observation_blocks", "observation_head"),
        "2b SphericalHistoryResponse": ("observation_history",),
        "2  observation gates": ("observation_branch_logits",),
        "3  multi-resolution residual (lift, 3 scales, fusion, head)": (
            "multiscale_lift", "multiscale_fine", "multiscale_mid", "multiscale_coarse",
            "multiscale_fusion", "multiscale_head",
        ),
        "3  multiscale gates": ("multiscale_logits",),
    }
    rows = []
    for label, prefixes in groups.items():
        count = sum(
            p.numel() for name, p in model.named_parameters()
            if name.split(".")[0] in prefixes
        )
        rows.append((label, count))
    rows.append(("total", sum(p.numel() for p in model.parameters())))
    return rows


if __name__ == "__main__":
    model = SphereDyn(54, 54, 5)
    print(model)
    for label, count in parameter_breakdown(model):
        print(f"{count:>12,d}  {label}")
