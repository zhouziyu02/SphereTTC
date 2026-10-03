from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_harmonics import InverseRealSHT, RealSHT

from src.spherettc import _complex_delta_clamp


@dataclass(frozen=True)
class SphereTTCv4Config:
    """Configuration for multi-timescale guarded spherical calibration."""

    lmax: int = 24
    mmax: int = 24
    n_bands: int = 4
    memory_size: int = 128
    min_memory: int = 24
    short_memory_size: int = 32
    short_half_life: float = 12.0
    long_half_life: float = 64.0
    expert_mode: str = "mixture"
    holdout_fraction: float = 0.25
    min_holdout: int = 8
    ridge: float = 0.05
    mixture_ridge: float = 0.05
    confidence_z: float = 0.5
    risk_floor: float = 0.01
    robust_quantile_scale: float = 3.0
    max_scale_delta: float = 0.5
    max_blend: float = 1.0
    coefficient_fraction: float = 0.25
    strength: float = 1.0
    taper_power: float = 1.0


class SphereTTCv4Calibrator:
    """Causal mixture of short- and long-memory spherical experts.

    Both experts fit band-shared complex affine corrections from an older
    chronological memory segment. A newer verified segment learns a
    non-negative per-variable/per-band mixture and rejects unsupported
    corrections. The experts are then refit using all eligible observations.
    The runner remains responsible for delayed-target eligibility.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCv4Config,
        grid: str = "equiangular",
    ):
        self.config = config
        if config.expert_mode not in {"mixture", "short", "long"}:
            raise ValueError(
                "expert_mode must be one of: mixture, short, long"
            )
        self.device = device
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.lmax = min(int(config.lmax), self.nlat)
        self.mmax = min(int(config.mmax), self.nlon // 2 + 1, self.lmax)
        self.sht = RealSHT(
            self.nlat,
            self.nlon,
            lmax=self.lmax,
            mmax=self.mmax,
            grid=grid,
        ).to(device)
        self.isht = InverseRealSHT(
            self.nlat,
            self.nlon,
            lmax=self.lmax,
            mmax=self.mmax,
            grid=grid,
        ).to(device)

        ell = torch.arange(self.lmax, dtype=torch.float32, device=device)
        denom = float(max(1, self.lmax - 1))
        taper = torch.cos(0.5 * torch.pi * ell / denom).clamp_min(0.0)
        self.taper = taper.pow(float(config.taper_power)).view(1, self.lmax, 1)

        n_bands = max(1, min(int(config.n_bands), self.lmax))
        self.band_index = torch.div(
            torch.arange(self.lmax, device=device) * n_bands,
            self.lmax,
            rounding_mode="floor",
        ).clamp_max(n_bands - 1)
        self.band_masks = torch.stack(
            [
                (self.band_index == band).to(torch.float32)
                for band in range(n_bands)
            ],
            dim=0,
        )
        self.n_bands = n_bands

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        """Encode (..., latitude, longitude) real fields."""
        return self.sht(torch.nan_to_num(field.float()))

    def _sample_weights(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
        half_life: float,
    ) -> torch.Tensor:
        """Return recency × residual-outlier weights."""
        n = pred_coeff.shape[0]
        age = torch.arange(
            n - 1,
            -1,
            -1,
            dtype=torch.float32,
            device=self.device,
        )
        recency = torch.exp2(-age / max(float(half_life), 1e-6))
        residual_energy = (target_coeff - pred_coeff).abs().square().mean(
            dim=(-2, -1)
        )
        median = residual_energy.median(dim=0).values.clamp_min(1e-12)
        cap = float(self.config.robust_quantile_scale) * median
        robust = (cap.unsqueeze(0) / residual_energy.clamp_min(1e-12)).clamp(
            max=1.0
        )
        return (recency[:, None] * robust).unsqueeze(-1).unsqueeze(-1)

    def _fit(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
        half_life: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit one band-shared/individual-shrinkage complex affine expert."""
        weights = self._sample_weights(
            pred_coeff,
            target_coeff,
            half_life,
        )
        weight_sum = weights.sum(dim=0).clamp_min(1e-12)
        mu_x = (weights * pred_coeff).sum(dim=0) / weight_sum
        mu_y = (weights * target_coeff).sum(dim=0) / weight_sum
        x_centered = pred_coeff - mu_x.unsqueeze(0)
        y_centered = target_coeff - mu_y.unsqueeze(0)
        var_x = (weights * x_centered.abs().square()).sum(dim=0)
        cov_yx = (
            weights * y_centered * x_centered.conj()
        ).sum(dim=0)

        coefficient_reference = var_x.mean(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-12)
        coefficient_ridge = float(self.config.ridge) * coefficient_reference
        coefficient_scale = (cov_yx + coefficient_ridge) / (
            var_x + coefficient_ridge
        )

        band_scales = []
        for band in range(self.n_bands):
            mask = self.band_masks[band][None, :, None]
            band_var = (var_x * mask).sum(dim=(-2, -1))
            band_cov = (cov_yx * mask).sum(dim=(-2, -1))
            band_ridge = float(self.config.ridge) * band_var.clamp_min(1e-12)
            band_scales.append(
                (band_cov + band_ridge)
                / (band_var + band_ridge).clamp_min(1e-12)
            )
        band_scale = torch.stack(band_scales, dim=1)
        shared_scale = band_scale[:, self.band_index].unsqueeze(-1)
        fraction = min(
            1.0,
            max(0.0, float(self.config.coefficient_fraction)),
        )
        scale = (
            fraction * coefficient_scale
            + (1.0 - fraction) * shared_scale
        )
        scale = _complex_delta_clamp(scale, self.config.max_scale_delta)
        bias = mu_y - scale * mu_x
        return scale, bias

    @staticmethod
    def _band_sample_energy(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return (value * mask[None, None, :, None]).sum(dim=(2, 3))

    def _expert_direction(
        self,
        pred: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return self.taper * (
            (scale - 1.0).unsqueeze(0) * pred + bias.unsqueeze(0)
        )

    def _select_experts(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Learn guarded short/long weights shaped (expert, variable, band)."""
        n = pred_coeff.shape[0]
        holdout = max(
            int(self.config.min_holdout),
            int(round(n * float(self.config.holdout_fraction))),
        )
        split = n - holdout
        short_size = min(int(self.config.short_memory_size), split)
        minimum_fit = max(2, int(self.config.min_memory) // 2)
        if split < minimum_fit or short_size < minimum_fit or holdout < 2:
            return torch.zeros(
                (2, pred_coeff.shape[1], self.n_bands),
                dtype=torch.float32,
                device=self.device,
            )

        short_pred = pred_coeff[split - short_size : split]
        short_target = target_coeff[split - short_size : split]
        short_scale, short_bias = self._fit(
            short_pred,
            short_target,
            self.config.short_half_life,
        )
        long_scale, long_bias = self._fit(
            pred_coeff[:split],
            target_coeff[:split],
            self.config.long_half_life,
        )
        raw = pred_coeff[split:]
        target = target_coeff[split:]
        error = raw - target
        short_direction = self._expert_direction(
            raw,
            short_scale,
            short_bias,
        )
        long_direction = self._expert_direction(
            raw,
            long_scale,
            long_bias,
        )

        selected_short = []
        selected_long = []
        maximum = float(self.config.max_blend)
        mixture_ridge = float(self.config.mixture_ridge)
        for band in range(self.n_bands):
            mask = self.band_masks[band]
            short_energy_sample = self._band_sample_energy(
                short_direction.abs().square(),
                mask,
            )
            long_energy_sample = self._band_sample_energy(
                long_direction.abs().square(),
                mask,
            )
            cross_sample = self._band_sample_energy(
                (short_direction.conj() * long_direction).real,
                mask,
            )
            rhs_short_sample = -self._band_sample_energy(
                (error.conj() * short_direction).real,
                mask,
            )
            rhs_long_sample = -self._band_sample_energy(
                (error.conj() * long_direction).real,
                mask,
            )
            short_energy = short_energy_sample.sum(dim=0)
            long_energy = long_energy_sample.sum(dim=0)
            cross = cross_sample.sum(dim=0)
            rhs_short = rhs_short_sample.sum(dim=0)
            rhs_long = rhs_long_sample.sum(dim=0)
            short_regularized = (
                (1.0 + mixture_ridge) * short_energy
            ).clamp_min(1e-12)
            long_regularized = (
                (1.0 + mixture_ridge) * long_energy
            ).clamp_min(1e-12)
            determinant = (
                short_regularized * long_regularized - cross.square()
            ).clamp_min(1e-12)
            joint_short = (
                rhs_short * long_regularized - rhs_long * cross
            ) / determinant
            joint_long = (
                rhs_long * short_regularized - rhs_short * cross
            ) / determinant
            short_only = rhs_short / short_regularized
            long_only = rhs_long / long_regularized
            zeros = torch.zeros_like(short_only)
            candidate_short = torch.stack(
                [
                    zeros,
                    short_only.clamp(0.0, maximum),
                    zeros,
                    joint_short.clamp(0.0, maximum),
                ],
                dim=0,
            )
            candidate_long = torch.stack(
                [
                    zeros,
                    zeros,
                    long_only.clamp(0.0, maximum),
                    joint_long.clamp(0.0, maximum),
                ],
                dim=0,
            )
            candidate_correction = (
                candidate_short[:, None, :, None, None]
                * short_direction[None]
                + candidate_long[:, None, :, None, None]
                * long_direction[None]
            )
            candidate_energy = self._band_sample_energy(
                (error[None] + candidate_correction).abs().square().flatten(
                    0,
                    1
                ),
                mask,
            ).reshape(4, holdout, -1).sum(dim=1)
            if self.config.expert_mode == "short":
                candidate_energy[2] = torch.inf
                candidate_energy[3] = torch.inf
            elif self.config.expert_mode == "long":
                candidate_energy[1] = torch.inf
                candidate_energy[3] = torch.inf
            best = candidate_energy.argmin(dim=0)
            gather_index = best.unsqueeze(0)
            best_short = candidate_short.gather(0, gather_index).squeeze(0)
            best_long = candidate_long.gather(0, gather_index).squeeze(0)

            correction = (
                best_short[None, :, None, None] * short_direction
                + best_long[None, :, None, None] * long_direction
            )
            raw_energy = self._band_sample_energy(
                error.abs().square(),
                mask,
            )
            corrected_energy = self._band_sample_energy(
                (error + correction).abs().square(),
                mask,
            )
            improvement = raw_energy - corrected_energy
            mean_gain = improvement.mean(dim=0)
            standard_error = improvement.std(dim=0, unbiased=False) / (
                float(holdout) ** 0.5
            )
            lower_bound = (
                mean_gain
                - float(self.config.confidence_z) * standard_error
            )
            raw_reference = raw_energy.mean(dim=0).clamp_min(1e-12)
            reliability = (
                lower_bound
                / (
                    mean_gain.abs()
                    + float(self.config.risk_floor) * raw_reference
                ).clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            selected_short.append(best_short * reliability)
            selected_long.append(best_long * reliability)

        strength = float(self.config.strength)
        return (
            torch.stack(
                [
                    torch.stack(selected_short, dim=1),
                    torch.stack(selected_long, dim=1),
                ],
                dim=0,
            )
            * strength
        )

    def calibrate_coefficients(
        self,
        current_coeff: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calibrate current coefficients using only verified memory."""
        n = pred_memory_coeff.shape[0]
        if n < int(self.config.min_memory):
            gate = torch.zeros(
                current_coeff.shape[0],
                dtype=torch.float32,
                device=self.device,
            )
            return current_coeff, gate

        expert_weights = self._select_experts(
            pred_memory_coeff,
            target_memory_coeff,
        )
        short_size = min(int(self.config.short_memory_size), n)
        short_scale, short_bias = self._fit(
            pred_memory_coeff[-short_size:],
            target_memory_coeff[-short_size:],
            self.config.short_half_life,
        )
        long_scale, long_bias = self._fit(
            pred_memory_coeff,
            target_memory_coeff,
            self.config.long_half_life,
        )
        short_direction = self.taper * (
            (short_scale - 1.0) * current_coeff + short_bias
        )
        long_direction = self.taper * (
            (long_scale - 1.0) * current_coeff + long_bias
        )
        short_gate = expert_weights[0, :, self.band_index].unsqueeze(-1)
        long_gate = expert_weights[1, :, self.band_index].unsqueeze(-1)
        correction = (
            short_gate * short_direction + long_gate * long_direction
        )
        summary_gate = expert_weights.sum(dim=0).mean(dim=1)
        return current_coeff + correction, summary_gate

    def calibrate(
        self,
        current_field: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(current_field)
        current_safe = torch.nan_to_num(current_field.float())
        current_coeff = self.encode(current_safe)
        calibrated_coeff, gate = self.calibrate_coefficients(
            current_coeff,
            pred_memory_coeff,
            target_memory_coeff,
        )
        correction = self.isht(calibrated_coeff - current_coeff)
        calibrated = current_safe + correction
        return torch.where(finite, calibrated, current_field), gate
