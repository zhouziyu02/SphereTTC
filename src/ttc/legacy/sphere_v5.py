from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch

from src.ttc.legacy.sphere_v4 import (
    SphereTTCv4Calibrator,
    SphereTTCv4Config,
)


@dataclass(frozen=True)
class SphereTTCv5Config:
    """Protected three-timescale spherical output correction."""

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


class SphereTTCv5Calibrator(SphereTTCv4Calibrator):
    """Causal multi-timescale affine experts with a protected no-op choice.

    Candidate active sets include no correction, every individual expert, all
    pairs, and the joint three-expert solution. The chronological holdout
    chooses the active set per variable and spherical band. A lower-confidence
    gain gate and worst-prefix regret gate shrink unsafe corrections to no-op.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        device: torch.device,
        config: SphereTTCv5Config,
        grid: str = "equiangular",
    ):
        self.v5_config = config
        super().__init__(
            nlat,
            nlon,
            device,
            SphereTTCv4Config(
                lmax=config.lmax,
                mmax=config.mmax,
                n_bands=config.n_bands,
                memory_size=config.memory_size,
                min_memory=config.min_memory,
                short_memory_size=config.short_memory_size,
                short_half_life=config.short_half_life,
                long_half_life=config.long_half_life,
                expert_mode="mixture",
                holdout_fraction=config.holdout_fraction,
                min_holdout=config.min_holdout,
                ridge=config.ridge,
                mixture_ridge=config.mixture_ridge,
                confidence_z=config.confidence_z,
                risk_floor=config.risk_floor,
                robust_quantile_scale=config.robust_quantile_scale,
                max_scale_delta=config.max_scale_delta,
                max_blend=config.max_blend,
                coefficient_fraction=config.coefficient_fraction,
                strength=config.strength,
                taper_power=config.taper_power,
            ),
            grid=grid,
        )

    def _expert_specs(self, available: int) -> list[tuple[int, float]]:
        cfg = self.v5_config
        return [
            (
                min(int(cfg.short_memory_size), available),
                float(cfg.short_half_life),
            ),
            (
                min(int(cfg.medium_memory_size), available),
                float(cfg.medium_half_life),
            ),
            (
                available,
                float(cfg.long_half_life),
            ),
        ]

    @staticmethod
    def _solve_active_set(
        gram: torch.Tensor,
        rhs: torch.Tensor,
        active: tuple[int, ...],
        ridge: float,
        maximum: float,
    ) -> torch.Tensor:
        """Solve one ridge active set for all variables."""
        variables, experts, _ = gram.shape
        result = torch.zeros(
            (variables, experts),
            dtype=gram.dtype,
            device=gram.device,
        )
        if not active:
            return result
        index = torch.as_tensor(
            active,
            dtype=torch.long,
            device=gram.device,
        )
        selected = gram.index_select(1, index).index_select(2, index)
        diagonal = torch.diagonal(selected, dim1=1, dim2=2)
        scale = diagonal.mean(dim=1).clamp_min(1e-12)
        eye = torch.eye(
            len(active),
            dtype=selected.dtype,
            device=selected.device,
        )[None]
        # Even a candidate with statistical ridge=0 needs scale-relative
        # numerical jitter: the three timescale experts can become exactly
        # collinear in a stationary band. An absolute 1e-8 is lost when Gram
        # energies are large, so retain a negligible relative floor.
        numerical_ridge = (
            (max(float(ridge), 0.0) + 1e-6) * scale + 1e-8
        )
        regularized = (
            selected + numerical_ridge[:, None, None] * eye
        )
        selected_rhs = rhs.index_select(1, index)
        rhs_column = selected_rhs.unsqueeze(-1)
        solution, info = torch.linalg.solve_ex(
            regularized,
            rhs_column,
            check_errors=False,
        )
        if torch.any(info != 0) or not torch.isfinite(solution).all():
            fallback = torch.linalg.pinv(regularized) @ rhs_column
            bad = (info != 0).view(-1, 1, 1) | ~torch.isfinite(
                solution
            ).all(dim=(1, 2), keepdim=True)
            solution = torch.where(bad, fallback, solution)
        solution = solution.squeeze(-1)
        solution = solution.clamp(0.0, maximum)
        result[:, index] = solution
        return result

    def _select_experts(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.v5_config
        n = pred_coeff.shape[0]
        holdout = max(
            int(cfg.min_holdout),
            int(round(n * float(cfg.holdout_fraction))),
        )
        split = n - holdout
        minimum_fit = max(2, int(cfg.min_memory) // 2)
        if split < minimum_fit or holdout < 2:
            return torch.zeros(
                (3, pred_coeff.shape[1], self.n_bands),
                dtype=torch.float32,
                device=self.device,
            )

        fit_pred = pred_coeff[:split]
        fit_target = target_coeff[:split]
        directions = []
        for size, half_life in self._expert_specs(split):
            if size < minimum_fit:
                return torch.zeros(
                    (3, pred_coeff.shape[1], self.n_bands),
                    dtype=torch.float32,
                    device=self.device,
                )
            scale, bias = self._fit(
                fit_pred[-size:],
                fit_target[-size:],
                half_life,
            )
            directions.append(
                self._expert_direction(
                    pred_coeff[split:],
                    scale,
                    bias,
                )
            )
        direction = torch.stack(directions, dim=0)
        error = pred_coeff[split:] - target_coeff[split:]
        selected_by_band = []
        active_sets = [
            active
            for count in range(4)
            for active in itertools.combinations(range(3), count)
        ]
        for band in range(self.n_bands):
            mask = self.band_masks[band]
            # direction: expert, sample, variable, ell, order
            gram = torch.empty(
                (pred_coeff.shape[1], 3, 3),
                dtype=torch.float32,
                device=self.device,
            )
            rhs = torch.empty(
                (pred_coeff.shape[1], 3),
                dtype=torch.float32,
                device=self.device,
            )
            for left in range(3):
                rhs[:, left] = -self._band_sample_energy(
                    (
                        error.conj()
                        * direction[left]
                    ).real,
                    mask,
                ).sum(dim=0)
                for right in range(3):
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
            candidate_error = (
                error.unsqueeze(0) + candidate_correction
            )
            candidate_energy = self._band_sample_energy(
                candidate_error.abs().square().flatten(0, 1),
                mask,
            ).reshape(
                len(active_sets),
                holdout,
                pred_coeff.shape[1],
            )
            total_energy = candidate_energy.sum(dim=1)
            best = total_energy.argmin(dim=0)
            # Select one candidate independently for each variable.  Put the
            # variable axis first so that gather does not accidentally index
            # every variable through position zero on the candidate tensor.
            best_weights = candidate_weights.permute(1, 0, 2).gather(
                1,
                best[:, None, None].expand(-1, 1, 3),
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

            prefix_regret = (
                corrected_energy - raw_energy
            ).cumsum(dim=0)
            raw_prefix = raw_energy.cumsum(dim=0).clamp_min(1e-12)
            worst_prefix_ratio = (
                prefix_regret.clamp_min(0.0) / raw_prefix
            ).max(dim=0).values
            tolerance = max(
                float(cfg.protection_tolerance),
                1e-8,
            )
            protection = (
                1.0 - worst_prefix_ratio / tolerance
            ).clamp(0.0, 1.0)
            selected_by_band.append(
                best_weights
                * reliability[:, None]
                * protection[:, None]
                * float(cfg.strength)
            )
        # expert, variable, band
        return torch.stack(selected_by_band, dim=2).permute(1, 0, 2)

    def calibrate_coefficients(
        self,
        current_coeff: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = pred_memory_coeff.shape[0]
        if n < int(self.v5_config.min_memory):
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
        summary_gate = weights.sum(dim=0).mean(dim=1)
        return current_coeff + correction, summary_gate
