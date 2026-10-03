from __future__ import annotations

from dataclasses import dataclass

import torch

from src.ttc.legacy.sphere_v5 import SphereTTCv5Config
from src.ttc.legacy.sphere_v6 import SphereTTCv6Calibrator, SphereTTCv6Config


@dataclass(frozen=True)
class SphereTTCv7Config:
    """Causal multi-timescale correction with a periodic residual anchor."""

    lmax: int = 64
    mmax: int = 64
    n_bands: int = 4
    memory_size: int = 400
    min_memory: int = 48
    short_memory_size: int = 48
    medium_memory_size: int = 128
    short_half_life: float = 16.0
    medium_half_life: float = 64.0
    long_half_life: float = 192.0
    use_seasonal_anchor: bool = True
    seasonal_half_life: float = 730.0
    seasonal_ridge: float = 0.1
    seasonal_harmonic_order: int = 2
    annual_period_days: float = 365.2425
    validation_blocks: int = 3
    worst_block_weight: float = 0.25
    holdout_fraction: float = 0.25
    min_holdout: int = 12
    ridge: float = 0.05
    mixture_ridge: float = 0.05
    confidence_z: float = 0.0
    risk_floor: float = 0.01
    robust_quantile_scale: float = 3.0
    max_scale_delta: float = 1.0
    max_blend: float = 1.0
    coefficient_fraction: float = 0.5
    strength: float = 1.0
    taper_power: float = 0.5
    protection_tolerance: float = 0.02


class SphereTTCv7Calibrator(SphereTTCv6Calibrator):
    """Protected causal spherical correction with seasonal residual memory.

    The first three experts are SphereTTC-v5's short, medium and long
    complex-affine corrections.  A fourth expert regresses verified spherical
    residual coefficients on annual and semi-annual harmonics.  A
    chronological holdout selects a non-negative mixture including the no-op
    active set.  The runner supplies *valid* dates and remains responsible for
    enforcing delayed-target eligibility.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCv7Config,
        grid: str = "equiangular",
    ):
        if int(config.seasonal_harmonic_order) not in {1, 2}:
            raise ValueError("seasonal_harmonic_order must be 1 or 2")
        if float(config.annual_period_days) <= 0:
            raise ValueError("annual_period_days must be positive")
        self.v7_config = config
        shared = {
            name: getattr(config, name)
            for name in SphereTTCv5Config.__dataclass_fields__
        }
        super().__init__(
            nlat,
            nlon,
            device,
            SphereTTCv6Config(
                **shared,
                use_bias_anchor=False,
                validation_blocks=config.validation_blocks,
                worst_block_weight=config.worst_block_weight,
            ),
            grid=grid,
        )

    def _seasonal_basis(self, valid_days: torch.Tensor) -> torch.Tensor:
        days = torch.as_tensor(
            valid_days,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        period = float(self.v7_config.annual_period_days)
        phase = (
            2.0
            * torch.pi
            * torch.remainder(days, period)
            / period
        )
        columns = [
            torch.ones_like(phase),
            torch.sin(phase),
            torch.cos(phase),
        ]
        if int(self.v7_config.seasonal_harmonic_order) == 2:
            columns.extend([torch.sin(2.0 * phase), torch.cos(2.0 * phase)])
        return torch.stack(columns, dim=1)

    def _validate_days(
        self,
        memory_valid_days: torch.Tensor,
        current_valid_day: torch.Tensor | float | int | None = None,
    ) -> torch.Tensor:
        days = torch.as_tensor(
            memory_valid_days,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if days.numel() > 1 and not torch.all(days[1:] > days[:-1]):
            raise ValueError("memory valid days must be strictly chronological")
        if current_valid_day is not None and days.numel():
            current = torch.as_tensor(
                current_valid_day,
                dtype=torch.float32,
                device=self.device,
            ).reshape(())
            if not bool(current > days[-1]):
                raise ValueError(
                    "current valid day must follow every verified memory target"
                )
        return days

    def _fit_seasonal(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
        fit_valid_days: torch.Tensor,
        apply_valid_days: torch.Tensor,
    ) -> torch.Tensor:
        """Fit and evaluate a robust periodic residual map."""
        design = self._seasonal_basis(fit_valid_days)
        apply_design = self._seasonal_basis(apply_valid_days)
        residual = target_coeff - pred_coeff
        weights = self._sample_weights(
            pred_coeff,
            target_coeff,
            self.v7_config.seasonal_half_life,
        ).squeeze(-1).squeeze(-1)

        gram = torch.einsum(
            "nv,np,nq->vpq",
            weights,
            design,
            design,
        )
        diagonal_scale = torch.diagonal(
            gram,
            dim1=1,
            dim2=2,
        ).mean(dim=1).clamp_min(1e-12)
        eye = torch.eye(
            design.shape[1],
            dtype=gram.dtype,
            device=self.device,
        )[None]
        ridge = (
            max(0.0, float(self.v7_config.seasonal_ridge))
            + 1e-6
        ) * diagonal_scale
        regularized = gram + ridge[:, None, None] * eye
        weighted_design = weights[:, :, None] * design[:, None, :]
        rhs = torch.einsum(
            "nvp,nvlm->vplm",
            weighted_design.to(residual.dtype),
            residual,
        )
        variables, basis_count, lmax, mmax = rhs.shape
        rhs_flat = rhs.reshape(variables, basis_count, lmax * mmax)
        matrix = regularized.to(rhs.dtype)
        solution, info = torch.linalg.solve_ex(
            matrix,
            rhs_flat,
            check_errors=False,
        )
        bad = (info != 0) | ~torch.isfinite(solution).all(dim=(1, 2))
        if torch.any(bad):
            fallback = torch.linalg.pinv(matrix) @ rhs_flat
            solution = torch.where(
                bad[:, None, None],
                fallback,
                solution,
            )
        coefficients = solution.reshape(
            variables,
            basis_count,
            lmax,
            mmax,
        )
        return torch.einsum(
            "np,vplm->nvlm",
            apply_design.to(coefficients.dtype),
            coefficients,
        )

    def _directions_v7(
        self,
        fit_pred: torch.Tensor,
        fit_target: torch.Tensor,
        apply_pred: torch.Tensor,
        fit_valid_days: torch.Tensor,
        apply_valid_days: torch.Tensor,
        minimum_fit: int,
    ) -> torch.Tensor | None:
        directions = []
        for size, half_life in self._expert_specs(fit_pred.shape[0]):
            if size < minimum_fit:
                return None
            scale, bias = self._fit(
                fit_pred[-size:],
                fit_target[-size:],
                half_life,
            )
            directions.append(
                self._expert_direction(apply_pred, scale, bias)
            )
        if self.v7_config.use_seasonal_anchor:
            seasonal = self._fit_seasonal(
                fit_pred,
                fit_target,
                fit_valid_days,
                apply_valid_days,
            )
            directions.append(self.taper * seasonal)
        return torch.stack(directions, dim=0)

    def _select_experts(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
        memory_valid_days: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.v7_config
        n = pred_coeff.shape[0]
        expert_count = 4 if cfg.use_seasonal_anchor else 3
        if memory_valid_days is None:
            if cfg.use_seasonal_anchor:
                raise ValueError("SphereTTC-v7 requires verified valid dates")
            days = torch.arange(n, dtype=torch.float32, device=self.device)
        else:
            days = self._validate_days(memory_valid_days)
        if days.numel() != n:
            raise ValueError("memory valid-date count does not match coefficients")
        holdout = max(
            int(cfg.min_holdout),
            int(round(n * float(cfg.holdout_fraction))),
        )
        split = n - holdout
        minimum_fit = max(2, int(cfg.min_memory) // 2)
        empty = torch.zeros(
            (expert_count, pred_coeff.shape[1], self.n_bands),
            dtype=torch.float32,
            device=self.device,
        )
        if split < minimum_fit or holdout < 2:
            return empty

        direction = self._directions_v7(
            pred_coeff[:split],
            target_coeff[:split],
            pred_coeff[split:],
            days[:split],
            days[split:],
            minimum_fit,
        )
        if direction is None:
            return empty
        error = pred_coeff[split:] - target_coeff[split:]
        active_sets = self._active_sets(expert_count)
        selected_by_band = []
        for band in range(self.n_bands):
            mask = self.band_masks[band]
            gram = torch.empty(
                (pred_coeff.shape[1], expert_count, expert_count),
                dtype=torch.float32,
                device=self.device,
            )
            rhs = torch.empty(
                (pred_coeff.shape[1], expert_count),
                dtype=torch.float32,
                device=self.device,
            )
            for left in range(expert_count):
                rhs[:, left] = -self._band_sample_energy(
                    (error.conj() * direction[left]).real,
                    mask,
                ).sum(dim=0)
                for right in range(expert_count):
                    gram[:, left, right] = self._band_sample_energy(
                        (
                            direction[left].conj()
                            * direction[right]
                        ).real,
                        mask,
                    ).sum(dim=0)
            candidate_weights = torch.stack(
                [
                    self._solve_active_set(
                        gram,
                        rhs,
                        active,
                        float(cfg.mixture_ridge),
                        float(cfg.max_blend),
                    )
                    for active in active_sets
                ],
                dim=0,
            )
            candidate_correction = torch.einsum(
                "kve,envlm->knvlm",
                candidate_weights.to(direction.dtype),
                direction,
            )
            candidate_energy = self._band_sample_energy(
                (
                    error.unsqueeze(0) + candidate_correction
                ).abs().square().flatten(0, 1),
                mask,
            ).reshape(len(active_sets), holdout, pred_coeff.shape[1])
            best = candidate_energy.sum(dim=1).argmin(dim=0)
            best_weights = candidate_weights.permute(1, 0, 2).gather(
                1,
                best[:, None, None].expand(-1, 1, expert_count),
            ).squeeze(1)
            best_correction = torch.einsum(
                "ve,envlm->nvlm",
                best_weights.to(direction.dtype),
                direction,
            )
            raw_energy = self._band_sample_energy(
                error.abs().square(),
                mask,
            )
            corrected_energy = self._band_sample_energy(
                (error + best_correction).abs().square(),
                mask,
            )
            improvement = raw_energy - corrected_energy
            mean_gain = improvement.mean(dim=0)
            standard_error = improvement.std(
                dim=0,
                unbiased=False,
            ) / (float(holdout) ** 0.5)
            lower_bound = mean_gain - float(cfg.confidence_z) * standard_error
            raw_reference = raw_energy.mean(dim=0).clamp_min(1e-12)
            reliability = (
                lower_bound
                / (
                    mean_gain.abs()
                    + float(cfg.risk_floor) * raw_reference
                ).clamp_min(1e-12)
            ).clamp(0.0, 1.0)
            block_reliability = self._block_reliability(
                improvement,
                mean_gain,
                raw_reference,
            )
            prefix_regret = (corrected_energy - raw_energy).cumsum(dim=0)
            raw_prefix = raw_energy.cumsum(dim=0).clamp_min(1e-12)
            worst_prefix_ratio = (
                prefix_regret.clamp_min(0.0) / raw_prefix
            ).max(dim=0).values
            tolerance = max(float(cfg.protection_tolerance), 1e-8)
            protection = (
                1.0 - worst_prefix_ratio / tolerance
            ).clamp(0.0, 1.0)
            selected_by_band.append(
                best_weights
                * reliability[:, None]
                * block_reliability[:, None]
                * protection[:, None]
                * float(cfg.strength)
            )
        return torch.stack(selected_by_band, dim=2).permute(1, 0, 2)

    def calibrate_coefficients(
        self,
        current_coeff: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
        memory_valid_days: torch.Tensor | None = None,
        current_valid_day: torch.Tensor | float | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = pred_memory_coeff.shape[0]
        expert_count = 4 if self.v7_config.use_seasonal_anchor else 3
        if memory_valid_days is None:
            if self.v7_config.use_seasonal_anchor:
                raise ValueError("SphereTTC-v7 requires verified valid dates")
            days = torch.arange(n, dtype=torch.float32, device=self.device)
        else:
            days = self._validate_days(
                memory_valid_days,
                current_valid_day,
            )
        if days.numel() != n:
            raise ValueError("memory valid-date count does not match coefficients")
        if n < int(self.v7_config.min_memory):
            gate = torch.zeros(
                current_coeff.shape[0],
                dtype=torch.float32,
                device=self.device,
            )
            return current_coeff, gate
        weights = self._select_experts(
            pred_memory_coeff,
            target_memory_coeff,
            days,
        )
        if weights.shape[0] != expert_count:
            raise RuntimeError("SphereTTC-v7 expert count mismatch")
        correction = torch.zeros_like(current_coeff)
        for expert_index, (size, half_life) in enumerate(
            self._expert_specs(n)
        ):
            scale, bias = self._fit(
                pred_memory_coeff[-size:],
                target_memory_coeff[-size:],
                half_life,
            )
            direction = self.taper * (
                (scale - 1.0) * current_coeff + bias
            )
            gate = weights[
                expert_index,
                :,
                self.band_index,
            ].unsqueeze(-1)
            correction = correction + gate * direction
        if self.v7_config.use_seasonal_anchor:
            if current_valid_day is None:
                raise ValueError("SphereTTC-v7 requires the current valid date")
            seasonal = self._fit_seasonal(
                pred_memory_coeff,
                target_memory_coeff,
                days,
                torch.as_tensor(
                    [current_valid_day],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )[0]
            gate = weights[3, :, self.band_index].unsqueeze(-1)
            correction = correction + gate * self.taper * seasonal
        summary_gate = weights.sum(dim=0).mean(dim=1)
        return current_coeff + correction, summary_gate
