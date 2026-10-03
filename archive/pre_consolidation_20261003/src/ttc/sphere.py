from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_harmonics import InverseRealSHT, RealSHT


@dataclass(frozen=True)
class SphereTTCConfig:
    """Hyperparameters for delay-aware spherical test-time calibration."""

    lmax: int = 16
    mmax: int = 16
    half_life: float = 32.0
    ridge: float = 0.05
    strength: float = 1.0
    min_memory: int = 24
    holdout_fraction: float = 0.25
    min_holdout: int = 8
    confidence_z: float = 0.5
    robust_quantile_scale: float = 3.0
    max_scale_delta: float = 0.5
    taper_power: float = 1.0


def _complex_delta_clamp(scale: torch.Tensor, max_delta: float) -> torch.Tensor:
    delta = scale - 1.0
    magnitude = delta.abs().clamp_min(1e-12)
    clipped = delta * (float(max_delta) / magnitude).clamp(max=1.0)
    return 1.0 + clipped


class SphereTTCCalibrator:
    """Causal spherical-harmonic affine calibration with risk control.

    The backbone is never modified.  Only low-order spherical coefficients are
    calibrated.  A chronological inner split estimates whether the proposed
    correction generalizes to more recent, already-verified forecasts.  The
    resulting gate prevents negative transfer before the correction is applied
    to the current forecast.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCConfig,
        grid: str = "equiangular",
    ):
        self.config = config
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

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        """Encode (..., lat, lon) real fields into truncated SHT coefficients."""
        return self.sht(torch.nan_to_num(field.float()))

    def _sample_weights(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Return recency and robust weights with shape (sample, variable, 1, 1)."""
        n = pred_coeff.shape[0]
        age = torch.arange(n - 1, -1, -1, dtype=torch.float32, device=self.device)
        recency = torch.exp2(-age / max(float(self.config.half_life), 1e-6))
        residual_energy = (target_coeff - pred_coeff).abs().square().mean(dim=(-2, -1))
        median = residual_energy.median(dim=0).values.clamp_min(1e-12)
        cap = float(self.config.robust_quantile_scale) * median
        robust = (cap.unsqueeze(0) / residual_energy.clamp_min(1e-12)).clamp(max=1.0)
        return (recency[:, None] * robust).unsqueeze(-1).unsqueeze(-1)

    def _fit(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._sample_weights(pred_coeff, target_coeff)
        weight_sum = weights.sum(dim=0).clamp_min(1e-12)
        mu_x = (weights * pred_coeff).sum(dim=0) / weight_sum
        mu_y = (weights * target_coeff).sum(dim=0) / weight_sum
        x_centered = pred_coeff - mu_x.unsqueeze(0)
        y_centered = target_coeff - mu_y.unsqueeze(0)
        var_x = (weights * x_centered.abs().square()).sum(dim=0) / weight_sum
        cov_yx = (weights * y_centered * x_centered.conj()).sum(dim=0) / weight_sum

        # Ridge toward the identity transformation, not toward zero.
        scale_ref = var_x.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        ridge = float(self.config.ridge) * scale_ref
        scale = (cov_yx + ridge) / (var_x + ridge)
        scale = _complex_delta_clamp(scale, self.config.max_scale_delta)
        bias = mu_y - scale * mu_x
        return scale, bias

    def _cross_fitted_gate(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate a conservative per-variable blend from a chronological split."""
        n = pred_coeff.shape[0]
        holdout = max(int(self.config.min_holdout), int(round(n * self.config.holdout_fraction)))
        split = n - holdout
        if split < int(self.config.min_memory) // 2 or holdout < 2:
            return torch.zeros(pred_coeff.shape[1], dtype=torch.float32, device=self.device)

        scale, bias = self._fit(pred_coeff[:split], target_coeff[:split])
        raw = pred_coeff[split:]
        target = target_coeff[split:]
        candidate = scale.unsqueeze(0) * raw + bias.unsqueeze(0)
        direction = candidate - raw
        raw_error = raw - target

        reduce_dims = (0, 2, 3)
        numerator = -(raw_error.conj() * direction).real.sum(dim=reduce_dims)
        denominator = direction.abs().square().sum(dim=reduce_dims).clamp_min(1e-12)
        optimal_blend = (numerator / denominator).clamp(0.0, 1.0)

        raw_mse_sample = raw_error.abs().square().mean(dim=(-2, -1))
        candidate_mse_sample = (candidate - target).abs().square().mean(dim=(-2, -1))
        improvement = raw_mse_sample - candidate_mse_sample
        mean_gain = improvement.mean(dim=0)
        standard_error = improvement.std(dim=0, unbiased=False) / (float(holdout) ** 0.5)
        lower_bound = mean_gain - float(self.config.confidence_z) * standard_error
        scale_ref = raw_mse_sample.mean(dim=0).clamp_min(1e-12)
        reliability = (lower_bound / (mean_gain.abs() + 0.01 * scale_ref)).clamp(0.0, 1.0)
        return optimal_blend * reliability * float(self.config.strength)

    def calibrate_coefficients(
        self,
        current_coeff: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = pred_memory_coeff.shape[0]
        n_variables = current_coeff.shape[0]
        if n < int(self.config.min_memory):
            gate = torch.zeros(n_variables, dtype=torch.float32, device=self.device)
            return current_coeff, gate
        gate = self._cross_fitted_gate(pred_memory_coeff, target_memory_coeff)
        scale, bias = self._fit(pred_memory_coeff, target_memory_coeff)
        candidate = scale * current_coeff + bias
        blend = gate[:, None, None] * self.taper
        return current_coeff + blend * (candidate - current_coeff), gate

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
