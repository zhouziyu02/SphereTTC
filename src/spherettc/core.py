"""SphereTTC: closed-form spherical harmonic test-time calibration.

The forecast backbone is fixed. Each lead maintains a separate chronological
memory, and each variable is calibrated independently. Verification timestamps
are configurable so archived target-date replay and completed-day replay can be
selected explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch_harmonics import InverseRealSHT, RealSHT


def eligible_from_buffer(buffer, current, memory_size):
    """Select the latest pairs whose verification-availability time has passed."""
    current = np.datetime64(current)
    eligible = [item for item in buffer if np.datetime64(item[0]) <= current]
    if memory_size is not None:
        eligible = eligible[-int(memory_size) :]
    return eligible


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
    """Closed-form spherical calibration of a frozen forecaster's outputs.

    Field shape: (..., latitude, longitude). Memory coefficient shape:
    (sample, variable, degree, order). The validation gate is per variable.
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

    @torch.no_grad()
    def encode(self, field: torch.Tensor) -> torch.Tensor:
        """Step (2): encode (..., lat, lon) real fields into truncated SHT coefficients."""
        return self.sht(torch.nan_to_num(field.float()))

    def _sample_weights(
        self,
        pred_coeff: torch.Tensor,
        target_coeff: torch.Tensor,
    ) -> torch.Tensor:
        """Step (3): recency and robust weights with shape (sample, variable, 1, 1)."""
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
        """Step (4): weighted affine fit per coefficient, ridge towards the identity."""
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
        """Step (5): estimate a conservative per-variable blend from a chronological split."""
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

    @torch.no_grad()
    def calibrate_coefficients(
        self,
        current_coeff: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps (4)-(6) in coefficient space; identity (gate 0) until min_memory pairs exist."""
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

    @torch.no_grad()
    def calibrate(
        self,
        current_field: torch.Tensor,
        pred_memory_coeff: torch.Tensor,
        target_memory_coeff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps (2)-(7) for one lead: (V, lat, lon) field -> calibrated field and gate (V,)."""
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

class SphereTTC:
    """Online calibration for one frozen forecasting backbone.

    Call ``calibrate(init_time, forecast)`` before ``update(init_time, forecast,
    truth)`` during replay. Fields have shape (lead, variable, latitude,
    longitude). Offline replay may store future targets; they remain ineligible
    until their verification-availability timestamp is reached.

    ``verification_delay_hours=0`` reproduces the archived target-date admission
    rule. Set it to 24 for UTC daily means that become available after the target
    day is complete; additional product latency can be added to that value.
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lead_hours: Sequence[int],
        config: SphereTTCConfig | None = None,
        memory_size: int = 64,
        device: str | torch.device = "cpu",
        init_step_hours: float = 24.0,
        verification_delay_hours: float = 0.0,
    ) -> None:
        delay = float(verification_delay_hours)
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("verification_delay_hours must be finite and nonnegative.")
        if not np.isfinite(init_step_hours) or init_step_hours <= 0:
            raise ValueError("init_step_hours must be finite and positive.")
        self.lead_hours = np.asarray(lead_hours, dtype=np.int64)
        if self.lead_hours.ndim != 1 or not self.lead_hours.size or np.any(self.lead_hours <= 0):
            raise ValueError("lead_hours must be a nonempty sequence of positive hours.")
        if int(memory_size) <= 0:
            raise ValueError("memory_size must be positive.")
        self.config = config or SphereTTCConfig()
        self.device = torch.device(device)
        self.calibrator = SphereTTCCalibrator(nlat, nlon, self.device, self.config)
        self.memory_size = int(memory_size)
        self.verification_delay_hours = delay
        self.verification_delay = np.timedelta64(int(round(delay * 3_600_000_000_000)), "ns")
        self.keep_size = (
            self.memory_size
            + int(np.ceil((float(self.lead_hours.max()) + delay) / float(init_step_hours)))
            + 4
        )
        self.buffers: list[list[tuple[np.datetime64, torch.Tensor, torch.Tensor]]] = [
            [] for _ in self.lead_hours
        ]
        self._last_calibration_time = None
        self._last_update_time = None

    def _as_tensor(self, array) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _check_field(self, field: torch.Tensor) -> None:
        if (
            field.ndim != 4
            or field.shape[0] != len(self.lead_hours)
            or tuple(field.shape[-2:]) != (self.calibrator.nlat, self.calibrator.nlon)
        ):
            raise ValueError("Fields must have shape (lead, variable, nlat, nlon).")

    @staticmethod
    def _check_time(init_time, previous):
        current = np.datetime64(init_time, "ns")
        if np.isnat(current):
            raise ValueError("Initialization time must not be NaT.")
        if previous is not None and current <= previous:
            raise ValueError("Initialization times must be strictly increasing.")
        return current

    @torch.no_grad()
    def calibrate(self, init_time, forecast):
        """Return calibrated fields, a (lead, variable) gate, and memory counts."""
        current = self._check_time(init_time, self._last_calibration_time)
        prediction = self._as_tensor(forecast)
        self._check_field(prediction)
        coefficients = self.calibrator.encode(prediction)
        outputs, gates, counts = [], np.zeros(prediction.shape[:2], np.float32), []
        for lead_index in range(len(self.lead_hours)):
            pred_current = prediction[lead_index]
            eligible = eligible_from_buffer(self.buffers[lead_index], current, self.memory_size)
            if not eligible:
                outputs.append(pred_current)
                counts.append(0)
                continue
            pred_memory = torch.stack([item[1] for item in eligible], dim=0)
            target_memory = torch.stack([item[2] for item in eligible], dim=0)
            current_coeff = coefficients[lead_index]
            calibrated_coeff, gate = self.calibrator.calibrate_coefficients(
                current_coeff, pred_memory, target_memory
            )
            correction = self.calibrator.isht(calibrated_coeff - current_coeff)
            current_safe = torch.nan_to_num(pred_current.float())
            outputs.append(
                torch.where(torch.isfinite(pred_current), current_safe + correction, pred_current)
            )
            gates[lead_index] = gate.detach().cpu().numpy()
            counts.append(len(eligible))
        self._last_calibration_time = current
        return torch.stack(outputs, dim=0), gates, counts

    @torch.no_grad()
    def update(self, init_time, forecast, truth) -> None:
        """Store raw forecast/target coefficients with delayed availability times."""
        current = self._check_time(init_time, self._last_update_time)
        prediction = self._as_tensor(forecast)
        target = self._as_tensor(truth)
        self._check_field(prediction)
        if prediction.shape != target.shape:
            raise ValueError("Forecast and truth shapes must match.")
        prediction_coeff = self.calibrator.encode(prediction)
        target_coeff = self.calibrator.encode(target)
        available_times = (
            current + self.lead_hours.astype("timedelta64[h]") + self.verification_delay
        )
        for lead_index, buffer in enumerate(self.buffers):
            buffer.append(
                (
                    available_times[lead_index],
                    prediction_coeff[lead_index].detach(),
                    target_coeff[lead_index].detach(),
                )
            )
            if len(buffer) > self.keep_size:
                self.buffers[lead_index] = buffer[-self.keep_size :]
        self._last_update_time = current
