from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_harmonics import InverseRealSHT, RealSHT

from src.spherettc import _complex_delta_clamp


@dataclass(frozen=True)
class SphereTTCv2Config:
    """Configuration for bandwise, componentwise risk-controlled SphereTTC."""

    lmax: int = 24
    mmax: int = 24
    half_life: float = 32.0
    ridge: float = 0.05
    strength: float = 1.0
    min_memory: int = 48
    holdout_fraction: float = 0.25
    validation_fraction: float = 0.5
    risk_fraction: float = 0.5
    min_holdout: int = 8
    confidence_z: float = 0.5
    robust_quantile_scale: float = 3.0
    max_scale_delta: float = 0.5
    taper_power: float = 1.0
    n_bands: int = 4
    gate_ridge: float = 0.05
    max_component_blend: float = 1.0
    risk_floor: float = 0.01
    hierarchy_weight: float = 0.25


class SphereTTCv2Calibrator:
    """Causal hierarchical spectral calibration with independent risk control.

    Historical samples are split chronologically into fit, blend-selection, and
    risk-estimation segments. Multiplicative and additive corrections receive
    separate non-negative gates for each variable and spherical-degree band.
    The most recent risk segment is not used to fit either the affine map or the
    component blend, preventing optimistic reuse of the same errors.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCv2Config,
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
        edges = torch.linspace(0, self.lmax, n_bands + 1, device=device)
        edges = torch.round(edges).to(torch.long)
        edges[0], edges[-1] = 0, self.lmax
        for index in range(1, edges.numel()):
            edges[index] = torch.maximum(edges[index], edges[index - 1] + 1)
        edges[-1] = self.lmax
        self.band_masks = torch.stack(
            [
                (
                    (torch.arange(self.lmax, device=device) >= edges[index])
                    & (torch.arange(self.lmax, device=device) < edges[index + 1])
                )
                for index in range(n_bands)
            ],
            dim=0,
        ).to(torch.float32)
        self.band_index = self.band_masks.argmax(dim=0)
        self.n_bands = n_bands

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        return self.sht(torch.nan_to_num(field.float()))

    def _sample_weights(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
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
        scale_ref = var_x.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        ridge = float(self.config.ridge) * scale_ref
        scale = (cov_yx + ridge) / (var_x + ridge)
        scale = _complex_delta_clamp(scale, self.config.max_scale_delta)
        bias = mu_y - scale * mu_x
        return scale, bias

    @staticmethod
    def _band_sum(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask[None, None, :, None]).sum(dim=(0, 2, 3))

    @staticmethod
    def _band_sample_sum(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask[None, None, :, None]).sum(dim=(2, 3))

    def _component_gates(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, n_variables = pred_coeff.shape[:2]
        validation = max(
            2 * int(self.config.min_holdout),
            int(round(n * float(self.config.validation_fraction))),
        )
        validation = min(validation, n - 2)
        risk_size = max(
            int(self.config.min_holdout),
            int(round(validation * float(self.config.risk_fraction))),
        )
        risk_size = min(risk_size, validation - 2)
        blend_size = validation - risk_size
        fit_end = n - validation
        blend_end = fit_end + blend_size
        if (
            fit_end < max(2, int(self.config.min_memory) // 2)
            or blend_size < 2
            or risk_size < 2
        ):
            zeros = torch.zeros(
                (n_variables, self.n_bands),
                dtype=torch.float32,
                device=self.device,
            )
            return zeros, zeros

        scale, bias = self._fit(pred_coeff[:fit_end], target_coeff[:fit_end])
        taper = self.taper

        blend_pred = pred_coeff[fit_end:blend_end]
        blend_target = target_coeff[fit_end:blend_end]
        blend_error = blend_pred - blend_target
        scale_direction = taper * (scale - 1.0).unsqueeze(0) * blend_pred
        bias_direction = taper * bias.unsqueeze(0).expand_as(blend_pred)

        scale_blends = []
        bias_blends = []
        for band in range(self.n_bands):
            mask = self.band_masks[band]
            gss = self._band_sum(scale_direction.abs().square(), mask)
            gbb = self._band_sum(bias_direction.abs().square(), mask)
            gsb = self._band_sum(
                (scale_direction.conj() * bias_direction).real,
                mask,
            )
            rhs_scale = -self._band_sum(
                (blend_error.conj() * scale_direction).real,
                mask,
            )
            rhs_bias = -self._band_sum(
                (blend_error.conj() * bias_direction).real,
                mask,
            )
            gss_regularized = gss * (1.0 + float(self.config.gate_ridge)) + 1e-12
            gbb_regularized = gbb * (1.0 + float(self.config.gate_ridge)) + 1e-12
            determinant = (
                gss_regularized * gbb_regularized - gsb.square()
            ).clamp_min(1e-12)
            scale_blend = (
                rhs_scale * gbb_regularized - rhs_bias * gsb
            ) / determinant
            bias_blend = (
                rhs_bias * gss_regularized - rhs_scale * gsb
            ) / determinant
            maximum = float(self.config.max_component_blend)
            scale_blends.append(scale_blend.clamp(0.0, maximum))
            bias_blends.append(bias_blend.clamp(0.0, maximum))
        scale_blend = torch.stack(scale_blends, dim=1)
        bias_blend = torch.stack(bias_blends, dim=1)

        risk_pred = pred_coeff[blend_end:]
        risk_target = target_coeff[blend_end:]
        raw_error = risk_pred - risk_target
        scale_direction = taper * (scale - 1.0).unsqueeze(0) * risk_pred
        bias_direction = taper * bias.unsqueeze(0).expand_as(risk_pred)
        reliabilities = []
        for band in range(self.n_bands):
            mask = self.band_masks[band]
            correction = (
                scale_blend[:, band][None, :, None, None] * scale_direction
                + bias_blend[:, band][None, :, None, None] * bias_direction
            )
            raw_energy = self._band_sample_sum(raw_error.abs().square(), mask)
            corrected_energy = self._band_sample_sum(
                (raw_error + correction).abs().square(),
                mask,
            )
            improvement = raw_energy - corrected_energy
            mean_gain = improvement.mean(dim=0)
            standard_error = improvement.std(dim=0, unbiased=False) / (
                float(risk_size) ** 0.5
            )
            lower_bound = mean_gain - float(self.config.confidence_z) * standard_error
            scale_ref = raw_energy.mean(dim=0).clamp_min(1e-12)
            reliability = (
                lower_bound
                / (
                    mean_gain.abs()
                    + float(self.config.risk_floor) * scale_ref
                ).clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            reliabilities.append(reliability)
        reliability = torch.stack(reliabilities, dim=1)
        strength = float(self.config.strength)
        return scale_blend * reliability * strength, bias_blend * reliability * strength

    def _global_gate(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Return the robust per-variable anchor used by the original method."""
        n = pred_coeff.shape[0]
        holdout = max(
            int(self.config.min_holdout),
            int(round(n * float(self.config.holdout_fraction))),
        )
        split = n - holdout
        if split < int(self.config.min_memory) // 2 or holdout < 2:
            return torch.zeros(
                pred_coeff.shape[1],
                dtype=torch.float32,
                device=self.device,
            )
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
        standard_error = improvement.std(dim=0, unbiased=False) / (
            float(holdout) ** 0.5
        )
        lower_bound = mean_gain - float(self.config.confidence_z) * standard_error
        scale_ref = raw_mse_sample.mean(dim=0).clamp_min(1e-12)
        reliability = (
            lower_bound
            / (
                mean_gain.abs()
                + float(self.config.risk_floor) * scale_ref
            )
        ).clamp(0.0, 1.0)
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
        global_gate = self._global_gate(pred_memory_coeff, target_memory_coeff)
        hierarchy_weight = min(1.0, max(0.0, float(self.config.hierarchy_weight)))
        if hierarchy_weight == 0.0:
            scale, bias = self._fit(pred_memory_coeff, target_memory_coeff)
            candidate = scale * current_coeff + bias
            blend = global_gate[:, None, None] * self.taper
            return (
                current_coeff + blend * (candidate - current_coeff),
                global_gate,
            )
        else:
            component_scale, component_bias = self._component_gates(
                pred_memory_coeff,
                target_memory_coeff,
            )
            anchor = global_gate[:, None]
            scale_gate = (
                (1.0 - hierarchy_weight) * anchor
                + hierarchy_weight * component_scale
            )
            bias_gate = (
                (1.0 - hierarchy_weight) * anchor
                + hierarchy_weight * component_bias
            )
        scale, bias = self._fit(pred_memory_coeff, target_memory_coeff)
        scale_by_degree = scale_gate[:, self.band_index].unsqueeze(-1)
        bias_by_degree = bias_gate[:, self.band_index].unsqueeze(-1)
        correction = self.taper * (
            scale_by_degree * (scale - 1.0) * current_coeff
            + bias_by_degree * bias
        )
        summary_gate = 0.5 * (
            scale_gate.mean(dim=1) + bias_gate.mean(dim=1)
        )
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
