from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_harmonics import InverseRealSHT, RealSHT

from src.spherettc import _complex_delta_clamp


@dataclass(frozen=True)
class SphereTTCv3Config:
    """Configuration for guarded band-shared spherical calibration."""

    lmax: int = 24
    mmax: int = 24
    n_bands: int = 4
    memory_size: int = 64
    min_memory: int = 24
    half_life: float = 32.0
    holdout_fraction: float = 0.25
    min_holdout: int = 8
    ridge: float = 0.05
    gate_ridge: float = 0.05
    confidence_z: float = 0.5
    risk_floor: float = 0.01
    robust_quantile_scale: float = 3.0
    max_scale_delta: float = 0.5
    max_blend: float = 1.0
    coefficient_fraction: float = 0.25
    strength: float = 1.0
    taper_power: float = 1.0


class SphereTTCv3Calibrator:
    """Causal spherical calibration with band sharing and a validation guard.

    The frozen backbone forecast is transformed to spherical harmonics. A
    complex multiplicative correction is shared within a small number of
    spherical-degree bands, while the additive bias remains coefficient
    specific. A chronological memory split fits the correction on older
    verified forecasts and accepts it only when it improves newer verified
    forecasts. No current or future verifying target is consumed here; causal
    memory eligibility is enforced by the runner.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCv3Config,
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

        n_bands = max(1, min(int(config.n_bands), self.lmax))
        # Integer division gives contiguous, non-empty and reproducible bands.
        band_index = torch.div(
            torch.arange(self.lmax, device=device) * n_bands,
            self.lmax,
            rounding_mode="floor",
        ).clamp_max(n_bands - 1)
        self.band_index = band_index
        self.band_masks = torch.stack(
            [(band_index == band).float() for band in range(n_bands)],
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
    ) -> torch.Tensor:
        """Return recency × outlier weights shaped (sample, variable, 1, 1)."""
        n = pred_coeff.shape[0]
        age = torch.arange(n - 1, -1, -1, dtype=torch.float32, device=self.device)
        recency = torch.exp2(-age / max(float(self.config.half_life), 1e-6))
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit a band-shared complex scale and coefficient-specific bias."""
        weights = self._sample_weights(pred_coeff, target_coeff)
        weight_sum = weights.sum(dim=0).clamp_min(1e-12)
        mu_x = (weights * pred_coeff).sum(dim=0) / weight_sum
        mu_y = (weights * target_coeff).sum(dim=0) / weight_sum
        x_centered = pred_coeff - mu_x.unsqueeze(0)
        y_centered = target_coeff - mu_y.unsqueeze(0)

        var_x = (weights * x_centered.abs().square()).sum(dim=0)
        cov_yx = (
            weights * y_centered * x_centered.conj()
        ).sum(dim=0)
        coefficient_reference = var_x.mean(dim=(-2, -1), keepdim=True).clamp_min(
            1e-12
        )
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
            band_scale = (band_cov + band_ridge) / (
                band_var + band_ridge
            ).clamp_min(1e-12)
            band_scales.append(band_scale)
        band_scale = torch.stack(band_scales, dim=1)
        shared_scale = band_scale[:, self.band_index].unsqueeze(-1)

        coefficient_fraction = min(
            1.0,
            max(0.0, float(self.config.coefficient_fraction)),
        )
        scale = (
            coefficient_fraction * coefficient_scale
            + (1.0 - coefficient_fraction) * shared_scale
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

    def _validation_gate(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate a non-negative per-variable, per-band validation blend."""
        n = pred_coeff.shape[0]
        holdout = max(
            int(self.config.min_holdout),
            int(round(n * float(self.config.holdout_fraction))),
        )
        split = n - holdout
        if split < max(2, int(self.config.min_memory) // 2) or holdout < 2:
            return torch.zeros(
                (pred_coeff.shape[1], self.n_bands),
                dtype=torch.float32,
                device=self.device,
            )

        scale, bias = self._fit(pred_coeff[:split], target_coeff[:split])
        raw = pred_coeff[split:]
        target = target_coeff[split:]
        direction = self.taper * (
            (scale - 1.0).unsqueeze(0) * raw + bias.unsqueeze(0)
        )
        raw_error = raw - target
        gates = []
        for band in range(self.n_bands):
            mask = self.band_masks[band]
            direction_energy = self._band_sample_energy(
                direction.abs().square(),
                mask,
            )
            numerator = -self._band_sample_energy(
                (raw_error.conj() * direction).real,
                mask,
            ).sum(dim=0)
            denominator = direction_energy.sum(dim=0)
            denominator = (
                (1.0 + float(self.config.gate_ridge)) * denominator
            ).clamp_min(1e-12)
            blend = (numerator / denominator).clamp(
                0.0,
                float(self.config.max_blend),
            )

            correction = blend[None, :, None, None] * direction
            raw_energy = self._band_sample_energy(
                raw_error.abs().square(),
                mask,
            )
            corrected_energy = self._band_sample_energy(
                (raw_error + correction).abs().square(),
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
            gates.append(blend * reliability)
        return torch.stack(gates, dim=1) * float(self.config.strength)

    def calibrate_coefficients(
        self,
        current_coeff: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calibrate current coefficients from already verified memory."""
        if pred_memory_coeff.shape[0] < int(self.config.min_memory):
            gate = torch.zeros(
                current_coeff.shape[0],
                dtype=torch.float32,
                device=self.device,
            )
            return current_coeff, gate
        gate = self._validation_gate(pred_memory_coeff, target_memory_coeff)
        scale, bias = self._fit(pred_memory_coeff, target_memory_coeff)
        gate_by_degree = gate[:, self.band_index].unsqueeze(-1)
        correction = self.taper * gate_by_degree * (
            (scale - 1.0) * current_coeff + bias
        )
        return current_coeff + correction, gate.mean(dim=1)

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
