from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch

from src.ttc.legacy.sphere_v5 import (
    SphereTTCv5Calibrator,
    SphereTTCv5Config,
)


@dataclass(frozen=True)
class SphereTTCv6Config:
    """Protected spherical correction with a conservative bias anchor."""

    lmax: int = 64
    mmax: int = 64
    n_bands: int = 4
    memory_size: int = 160
    min_memory: int = 24
    short_memory_size: int = 48
    medium_memory_size: int = 96
    short_half_life: float = 16.0
    medium_half_life: float = 48.0
    long_half_life: float = 112.0
    bias_anchor_half_life: float = 64.0
    use_bias_anchor: bool = True
    validation_blocks: int = 3
    worst_block_weight: float = 0.5
    holdout_fraction: float = 0.25
    min_holdout: int = 8
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


class SphereTTCv6Calibrator(SphereTTCv5Calibrator):
    """Four-expert causal correction with block-robust protection.

    SphereTTC-v5's short, medium and long complex-affine experts are augmented
    with a bias-only anchor. The anchor can capture a stable residual location
    without estimating a high-variance multiplicative response. Candidate
    mixtures retain the no-op expert. Their correction is additionally shrunk
    when gains do not persist across chronological holdout blocks.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCv6Config,
        grid: str = "equiangular",
    ):
        self.v6_config = config
        super().__init__(
            nlat,
            nlon,
            device,
            SphereTTCv5Config(
                **{
                    name: getattr(config, name)
                    for name in SphereTTCv5Config.__dataclass_fields__
                }
            ),
            grid=grid,
        )

    def _fit_bias(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
        half_life: float,
    ) -> torch.Tensor:
        weights = self._sample_weights(
            pred_coeff,
            target_coeff,
            half_life,
        )
        residual = target_coeff - pred_coeff
        return (
            (weights * residual).sum(dim=0)
            / weights.sum(dim=0).clamp_min(1e-12)
        )

    def _directions(
        self,
        fit_pred: torch.Tensor,
        fit_target: torch.Tensor,
        apply_pred: torch.Tensor,
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
        if self.v6_config.use_bias_anchor:
            anchor = self._fit_bias(
                fit_pred,
                fit_target,
                self.v6_config.bias_anchor_half_life,
            )
            directions.append(
                self.taper * anchor.unsqueeze(0).expand_as(apply_pred)
            )
        return torch.stack(directions, dim=0)

    @staticmethod
    def _active_sets(experts: int) -> list[tuple[int, ...]]:
        return [
            active
            for count in range(experts + 1)
            for active in itertools.combinations(range(experts), count)
        ]

    def _block_reliability(
        self,
        improvement: torch.Tensor,
        mean_gain: torch.Tensor,
        raw_reference: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.v6_config
        blocks = max(
            1,
            min(int(cfg.validation_blocks), improvement.shape[0]),
        )
        block_means = torch.stack(
            [
                part.mean(dim=0)
                for part in torch.tensor_split(
                    improvement,
                    blocks,
                    dim=0,
                )
                if part.shape[0]
            ],
            dim=0,
        )
        worst = block_means.min(dim=0).values
        robust = (
            worst
            / (
                mean_gain.abs()
                + float(cfg.risk_floor) * raw_reference
            ).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        weight = min(1.0, max(0.0, float(cfg.worst_block_weight)))
        return (1.0 - weight) + weight * robust

    def _select_experts(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.v6_config
        n = pred_coeff.shape[0]
        expert_count = 4 if cfg.use_bias_anchor else 3
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

        direction = self._directions(
            pred_coeff[:split],
            target_coeff[:split],
            pred_coeff[split:],
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
            ).reshape(
                len(active_sets),
                holdout,
                pred_coeff.shape[1],
            )
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
            lower_bound = (
                mean_gain
                - float(cfg.confidence_z) * standard_error
            )
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
            prefix_regret = (
                corrected_energy - raw_energy
            ).cumsum(dim=0)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = pred_memory_coeff.shape[0]
        expert_count = 4 if self.v6_config.use_bias_anchor else 3
        if n < int(self.v6_config.min_memory):
            gate = torch.zeros(
                current_coeff.shape[0],
                dtype=torch.float32,
                device=self.device,
            )
            return current_coeff, gate
        weights = self._select_experts(
            pred_memory_coeff,
            target_memory_coeff,
        )
        if weights.shape[0] != expert_count:
            raise RuntimeError("SphereTTC-v6 expert count mismatch")
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
        if self.v6_config.use_bias_anchor:
            anchor = self._fit_bias(
                pred_memory_coeff,
                target_memory_coeff,
                self.v6_config.bias_anchor_half_life,
            )
            gate = weights[3, :, self.band_index].unsqueeze(-1)
            correction = correction + gate * self.taper * anchor
        summary_gate = weights.sum(dim=0).mean(dim=1)
        return current_coeff + correction, summary_gate
