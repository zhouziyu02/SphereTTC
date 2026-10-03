"""SphereTTC: spherical test-time calibration for medium-range weather forecasting.

This single file contains the complete SphereTTC module of the paper.

Setting
-------
A frozen forecasting backbone issues, at every initialization time t, daily-mean
forecasts  y_hat_t(tau)  for leads tau in {24, 72, 120, 168, 240} h and V variables
on the 1.5 deg sphere (121 x 240).  The verification y_t(tau) of a forecast is only
available after its valid time t + tau (delayed supervision).  SphereTTC never
updates the backbone; it post-processes y_hat_t(tau) using only forecast /
verification pairs that are already verified at time t.

Mode A -- spherical calibration of a single backbone        (paper Result A, 11 backbones)
------------------------------------------------------------------------------------------
For every lead tau and every variable v independently, at every initialization t:

  (1) Delayed-verification memory    M_t(tau): the last K = memory_size pairs
                                     (y_hat_s(tau), y_s(tau)) with valid time s + tau <= t
                                     [Part 1: eligible_from_buffer]
  (2) Spherical-harmonic encoding    a = SHT(y_hat), b = SHT(y), truncated to l < lmax, m < mmax
                                     [SphereTTCCalibrator.encode]
  (3) Sample weights                 w_k = recency 2^(-age_k / half_life)
                                           x robust min(1, robust_quantile_scale * median(e) / e_k),
                                     e_k = spectral residual energy of pair k     [_sample_weights]
  (4) Conservative affine fit        b ~ s * a + beta per coefficient (l, m), weighted ridge
                                     regression shrunk towards s = 1; |s - 1| <= max_scale_delta
                                     [_fit]
  (5) Chronological validation gate  refit (4) on the older 75 % of M, evaluate on the newest
                                     25 %: gate = clip(optimal blend, 0, 1)
                                                  x reliability(lower confidence bound of the
                                                    MSE improvement, confidence_z)
                                                  x strength
                                     gate = 0  =>  no-op, the forecast is returned unchanged
                                     [_cross_fitted_gate]
  (6) Spectral taper                 a_cal = a + gate * taper(l) * (s * a + beta - a),
                                     taper(l) = cos(pi/2 * l / (lmax - 1)) ** taper_power
                                     [calibrate_coefficients]
  (7) Inverse transform              y_tilde = y_hat + ISHT(a_cal - a)          [calibrate]

Online use: :class:`SphereTTC` (Part 3) wraps (1)-(7) as a plug-in for any frozen
backbone.  The experiment runner ``scripts/run_ttc.py --method sphere_ttc`` executes
the same steps (it imports Parts 1-2 from this file); the equivalence is tested in
``tests/test_consolidation_equivalence.py``.

Frozen hyper-parameters (selected on 2017 only):
  * 10 backbones (publication setting): memory 64 (CirT 32), half_life 32 (CirT 16),
    lmax = mmax 40 for the 6 locally trained backbones (CirT 32) and 24 for the
    4 released systems; ridge 0.05, strength 1.0, min_memory 24, holdout 0.25,
    min_holdout 8, confidence_z 0.5, robust_quantile_scale 3, max_scale_delta 0.5,
    taper_power 1  -> artifacts/ttc_publication_corrected_20260726/params/*.json
  * GraphCast: global setting ``l64_s08`` (lmax = mmax 64, strength 0.8, otherwise
    as above) -> artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json

Mode B -- multi-provider constrained combination            (paper Result B, SphereDyn)
------------------------------------------------------------------------------------------
"SphereTTC-v22".  When several forecast providers are available, the target backbone
f_0 is combined with R released reference systems f_1..f_R per (lead, variable):

    y_tilde = alpha * f_0 + (1 - alpha) * sum_r omega_r f_r + c - S(lat, lon)

  * backbone anchor alpha in {0.5, 0.75} (or the raw no-op alpha = 1): the target
    backbone always keeps at least half of the weight;
  * reference weights omega (summing to 1) are one of: minimum-error-variance ridge
    weights (9 ridge ratios), uniform weights, or a single reference (one-hot);
  * c: global bias = -(mean combined error); S: spatial bias map, shrunk by a factor
    in {0, .25, .5, .75, 1};
  * the candidate (and the shrinkage) is chosen by error energy on a chronological
    validation split of 2017 (last 121 days), then refit on all of 2017 and frozen;
    there is no online update and no spherical-harmonic step in this mode.
  In the frozen SphereDyn weights alpha = 0.5 in all 245 (lead, variable) cells and the
  references are FourCastNetV2, OneForecast, FuXi, Pangu, GraphCast.
  [Part 4: fit_constrained_combination / apply_constrained_combination; the 2017
  statistics are accumulated by scripts/spheredyn/run_main_spherettc_v2.py::_fit]

Not part of the final module
----------------------------
The multi-timescale / seasonal calibration experts of the project proposal exist
only as experimental variants (src/ttc/legacy/sphere_v2..v7.py) and are not used by
any frozen result.

Causality note (docs/04_AUDIT_AND_RISKS.md, R1)
----------------------------------------------
The admission rule in ``eligible_from_buffer`` is ``valid_time <= t``.  For daily-mean
targets of 00 UTC-initialized systems the verification day is only complete at
t + 18 h; the planned fix is ``valid_time + 1 day <= t`` in this one function.

Provenance
----------
Parts 1, 2 and 4 are moved verbatim (identical code, docstrings extended) from
src/ttc/memory.py, scripts/run_ttc.py::_eligible_from_buffer, src/ttc/sphere.py and
scripts/spheredyn/evaluate_goal_reference_ensemble.py (originals archived in
archive/pre_consolidation_20261003/); ``verify_migration.py`` checks the code is
AST-identical.  Part 3 and the public wrappers of Part 4 are new and tested against
the experiment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch_harmonics import InverseRealSHT, RealSHT


# =============================================================================
# Part 1 -- delayed-verification memory (causality)
# =============================================================================


def eligible_from_buffer(buffer, current, memory_size):
    """Return the last ``memory_size`` buffer items whose verification time is <= ``current``.

    ``buffer`` holds ``(valid_time, prediction, target)`` items in chronological
    order.  This is the single causal admission rule used by scripts/run_ttc.py
    for every online method (see the causality note in the module docstring).
    """
    current = np.datetime64(current)
    eligible = [item for item in buffer if np.datetime64(item[0]) <= current]
    if memory_size is not None:
        eligible = eligible[-int(memory_size) :]
    return eligible


@dataclass
class EligibleMemory:
    """Index-based view of the same causal rule (used by unit tests)."""

    max_size: int | None = None
    mode: str = "per_lead"

    def get_eligible_indices(self, current_init_time, init_times, valid_times, lead_index: int | None = None):
        current = np.datetime64(current_init_time)
        init_times = np.asarray(init_times, dtype="datetime64[ns]")
        valid_times = np.asarray(valid_times, dtype="datetime64[ns]")
        if valid_times.ndim != 2:
            raise ValueError("valid_times must have shape [N, L].")
        if self.mode == "per_lead":
            if lead_index is None:
                raise ValueError("lead_index is required for per_lead memory.")
            eligible = np.where(valid_times[:, int(lead_index)] <= current)[0]
            eligible = eligible[init_times[eligible] < current]
            if self.max_size is not None:
                eligible = eligible[-int(self.max_size) :]
            return eligible
        if self.mode == "cross_lead":
            sample_idx, lead_idx = np.where(valid_times <= current)
            keep = init_times[sample_idx] < current
            pairs = list(zip(sample_idx[keep].tolist(), lead_idx[keep].tolist()))
            pairs.sort(key=lambda x: valid_times[x[0], x[1]])
            if self.max_size is not None:
                pairs = pairs[-int(self.max_size) :]
            return pairs
        raise ValueError(f"Unknown memory mode: {self.mode}")


def assert_no_future_targets(current_init_time, selected_valid_times):
    """Raise if any selected memory item has a verification time after ``current_init_time``."""
    current = np.datetime64(current_init_time)
    selected = np.asarray(selected_valid_times, dtype="datetime64[ns]")
    if selected.size and np.any(selected > current):
        raise AssertionError("TTC memory leakage: selected target valid_time is in the future.")


# =============================================================================
# Part 2 -- Mode A: spherical-harmonic affine calibrator, steps (2)-(7)
# =============================================================================


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

    Shapes: fields (..., lat, lon); coefficients (..., lmax, mmax) complex;
    memory coefficients (K, V, lmax, mmax); gate (V,).
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


# =============================================================================
# Part 3 -- Mode A as an online plug-in for any frozen backbone, steps (1)-(7)
# =============================================================================


class SphereTTC:
    """Plug-and-play online SphereTTC for one frozen forecasting backbone.

    Daily cycle (identical to ``scripts/run_ttc.py --method sphere_ttc``)::

        ttc = SphereTTC(121, 240, lead_hours=[24, 72, 120, 168, 240],
                        config=SphereTTCConfig(lmax=64, mmax=64, strength=0.8),
                        memory_size=64)
        for t in init_times:
            y_hat = backbone(...)                          # (L, V, lat, lon)
            y_tilde, gate, n_memory = ttc.calibrate(t, y_hat)
            ttc.update(t, y_hat, y_true)                   # verification of this forecast;
                                                           # admitted only once valid_time <= t'

    ``update`` may be called as soon as the truth is known (offline replay) or when
    it arrives (operations): admission is decided at calibration time from the
    valid times, so future information is never used.
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
    ) -> None:
        self.config = config or SphereTTCConfig()
        self.device = torch.device(device)
        self.calibrator = SphereTTCCalibrator(nlat, nlon, self.device, self.config)
        self.lead_hours = np.asarray(lead_hours, dtype=np.int64)
        self.memory_size = int(memory_size)
        # retain enough pairs: K eligible + those still awaiting verification
        self.keep_size = (
            self.memory_size
            + int(np.ceil(float(self.lead_hours.max()) / float(init_step_hours)))
            + 4
        )
        self.buffers: list[list[tuple[np.datetime64, torch.Tensor, torch.Tensor]]] = [
            [] for _ in self.lead_hours
        ]

    def _as_tensor(self, array) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    def calibrate(self, init_time, forecast):
        """Calibrate one forecast (L, V, lat, lon) issued at ``init_time``.

        Returns (calibrated (L, V, lat, lon) tensor, gate (L, V) array, memory size per lead).
        """
        current = np.datetime64(init_time, "ns")
        prediction = self._as_tensor(forecast)
        coefficients = self.calibrator.encode(prediction)  # all leads at once
        outputs, gates, counts = [], np.zeros(prediction.shape[:2], np.float32), []
        for lead_index in range(len(self.lead_hours)):
            pred_current = prediction[lead_index]
            eligible = eligible_from_buffer(self.buffers[lead_index], current, self.memory_size)
            if not eligible:
                outputs.append(pred_current)
                counts.append(0)
                continue
            assert_no_future_targets(current, [item[0] for item in eligible])
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
        return torch.stack(outputs, dim=0), gates, counts

    def update(self, init_time, forecast, truth) -> None:
        """Store the forecast/verification pair of ``init_time`` (raw forecast, not calibrated)."""
        prediction_coeff = self.calibrator.encode(self._as_tensor(forecast))
        target_coeff = self.calibrator.encode(self._as_tensor(truth))
        valid_times = np.datetime64(init_time, "ns") + self.lead_hours.astype("timedelta64[h]")
        for lead_index, buffer in enumerate(self.buffers):
            buffer.append(
                (
                    valid_times[lead_index],
                    prediction_coeff[lead_index].detach(),
                    target_coeff[lead_index].detach(),
                )
            )
            if len(buffer) > self.keep_size:
                self.buffers[lead_index] = buffer[-self.keep_size :]


# =============================================================================
# Part 4 -- Mode B: multi-provider constrained combination ("SphereTTC-v22")
# =============================================================================

REFERENCES = (
    "fourcastnetv2",
    "oneforecast",
    "fuxi",
    "pangu",
    "graphcast",
)
RIDGE_RATIOS = np.asarray(
    [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    dtype=np.float64,
)
SPATIAL_SHRINKAGES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])


def _ridge_weights(covariance: np.ndarray, ratio: float) -> np.ndarray:
    """Minimum-error-variance weights summing to one, with relative ridge ``ratio``."""
    features = covariance.shape[0]
    scale = max(float(np.trace(covariance)) / features, 1e-18)
    matrix = 0.5 * (covariance + covariance.T)
    matrix = matrix + (float(ratio) * scale + 1e-12 * scale) * np.eye(features)
    ones = np.ones(features)
    try:
        direction = np.linalg.solve(matrix, ones)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(matrix, rcond=1e-12) @ ones
    denominator = float(ones @ direction)
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.full(features, 1.0 / features)
    weights = direction / denominator
    if not np.all(np.isfinite(weights)):
        return np.full(features, 1.0 / features)
    return weights


def _energy(
    second_moment: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> float:
    """Mean squared combined error  E[(w.e + bias)^2]  from first and second moments."""
    return float(
        weights @ second_moment @ weights
        + 2.0 * bias * (weights @ mean)
        + bias * bias
    )


def _candidate_weights(
    covariance: np.ndarray,
    minimum_backbone_weight: float,
) -> list[tuple[str, int, float, np.ndarray]]:
    """Candidate combination weights; the target backbone is the last feature.

    With ``minimum_backbone_weight > 0`` (SphereTTC-v22): anchor alpha in
    {max(min, .5), max(min, .75)} on the backbone and (1 - alpha) times one of
    {ridge weights x 9 ratios, uniform, one-hot per reference} on the references,
    plus the raw no-op (alpha = 1).
    """
    features = covariance.shape[0]
    if minimum_backbone_weight > 0:
        reference_features = features - 1
        reference_covariance = covariance[:-1, :-1]
        reference_candidates: list[tuple[str, int, np.ndarray]] = [
            ("anchor_ridge", index, _ridge_weights(reference_covariance, ratio))
            for index, ratio in enumerate(RIDGE_RATIOS)
        ]
        reference_candidates.append(
            ("anchor_uniform", -1, np.full(reference_features, 1.0 / reference_features))
        )
        for index in range(reference_features):
            weights = np.zeros(reference_features)
            weights[index] = 1.0
            reference_candidates.append(("anchor_onehot", index, weights))
        anchors = sorted(
            {
                float(minimum_backbone_weight),
                max(float(minimum_backbone_weight), 0.5),
                max(float(minimum_backbone_weight), 0.75),
            }
        )
        candidates = []
        for kind, index, reference_weights in reference_candidates:
            for anchor in anchors:
                if anchor >= 1.0:
                    continue
                weights = np.zeros(features)
                weights[:-1] = (1.0 - anchor) * reference_weights
                weights[-1] = anchor
                candidates.append((kind, index, anchor, weights))
        raw = np.zeros(features)
        raw[-1] = 1.0
        candidates.append(("raw_noop", -1, 1.0, raw))
        return candidates
    candidates = [
        ("ridge", index, float("nan"), _ridge_weights(covariance, ratio))
        for index, ratio in enumerate(RIDGE_RATIOS)
    ]
    candidates.append(
        ("uniform", -1, float("nan"), np.full(features, 1.0 / features))
    )
    for index in range(features):
        weights = np.zeros(features)
        weights[index] = 1.0
        candidates.append(("one_hot", index, float("nan"), weights))
    return candidates


def _feature_statistics(
    statistics: dict[str, np.ndarray],
    backbone_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Assemble error cross-products/sums with the backbone as the last feature."""
    reference_count = statistics["reference_sum"].shape[-1]
    cross = np.zeros(
        statistics["reference_cross"].shape[:2]
        + (reference_count + 1, reference_count + 1),
        dtype=np.float64,
    )
    cross[..., :reference_count, :reference_count] = statistics[
        "reference_cross"
    ]
    cross[..., :reference_count, reference_count] = statistics[
        "raw_reference_cross"
    ][backbone_index]
    cross[..., reference_count, :reference_count] = statistics[
        "raw_reference_cross"
    ][backbone_index]
    cross[..., reference_count, reference_count] = statistics["raw_sse"][
        backbone_index
    ]
    sums = np.concatenate(
        [
            statistics["reference_sum"],
            statistics["raw_sum"][backbone_index, ..., None],
        ],
        axis=-1,
    )
    return cross, sums, float(statistics["count"])


def _fit_backbone(
    backbone_index: int,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    latitude: np.ndarray,
    minimum_backbone_weight: float,
) -> dict[str, np.ndarray]:
    """Select (on validation) and refit (on train + validation) the combination weights."""
    train_cross, train_sum, train_count = _feature_statistics(train, backbone_index)
    val_cross, val_sum, val_count = _feature_statistics(validation, backbone_index)
    full_cross = train_cross + val_cross
    full_sum = train_sum + val_sum
    full_count = train_count + val_count
    leads, variables, features = train_sum.shape

    selected_kind = np.empty((leads, variables), dtype="U16")
    selected_index = np.zeros((leads, variables), dtype=np.int16)
    selected_anchor = np.full((leads, variables), np.nan, dtype=np.float64)
    train_weights = np.zeros((leads, variables, features), dtype=np.float64)
    train_bias = np.zeros((leads, variables), dtype=np.float64)

    for lead in range(leads):
        for variable in range(variables):
            train_second = train_cross[lead, variable] / train_count
            train_mean = train_sum[lead, variable] / train_count
            train_covariance = train_second - np.outer(train_mean, train_mean)
            val_second = val_cross[lead, variable] / val_count
            val_mean = val_sum[lead, variable] / val_count
            best: tuple[float, str, int, float, np.ndarray, float] | None = None
            for kind, index, anchor, weights in _candidate_weights(
                train_covariance, minimum_backbone_weight
            ):
                bias = -float(weights @ train_mean)
                score = _energy(val_second, val_mean, weights, bias)
                candidate = (score, kind, index, anchor, weights, bias)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            assert best is not None
            _, kind, index, anchor, weights, bias = best
            selected_kind[lead, variable] = kind
            selected_index[lead, variable] = index
            selected_anchor[lead, variable] = anchor
            train_weights[lead, variable] = weights
            train_bias[lead, variable] = bias

    reference_count = train["reference_map_sum"].shape[-1]
    train_feature_map = np.concatenate(
        [
            train["reference_map_sum"],
            train["raw_map_sum"][backbone_index, ..., None],
        ],
        axis=-1,
    )
    val_feature_map = np.concatenate(
        [
            validation["reference_map_sum"],
            validation["raw_map_sum"][backbone_index, ..., None],
        ],
        axis=-1,
    )
    train_mean_map = (
        np.einsum("lvxym,lvm->lvxy", train_feature_map, train_weights, optimize=True)
        / float(train["days"])
        + train_bias[..., None, None]
    )
    val_sum_map = (
        np.einsum("lvxym,lvm->lvxy", val_feature_map, train_weights, optimize=True)
        + float(validation["days"]) * train_bias[..., None, None]
    )
    spatial_shrink = np.zeros((leads, variables), dtype=np.float64)
    latitude_map = latitude[None, None, :, None]
    linear = np.sum(latitude_map * train_mean_map * val_sum_map, axis=(2, 3))
    quadratic = float(validation["days"]) * np.sum(
        latitude_map * train_mean_map * train_mean_map,
        axis=(2, 3),
    )
    deltas = np.stack(
        [
            -2.0 * value * linear + value * value * quadratic
            for value in SPATIAL_SHRINKAGES
        ],
        axis=0,
    )
    spatial_shrink = SPATIAL_SHRINKAGES[np.argmin(deltas, axis=0)]

    final_weights = np.zeros_like(train_weights)
    final_bias = np.zeros_like(train_bias)
    for lead in range(leads):
        for variable in range(variables):
            full_second = full_cross[lead, variable] / full_count
            full_mean = full_sum[lead, variable] / full_count
            full_covariance = full_second - np.outer(full_mean, full_mean)
            kind = str(selected_kind[lead, variable])
            index = int(selected_index[lead, variable])
            anchor = float(selected_anchor[lead, variable])
            if kind == "anchor_ridge":
                reference_weights = _ridge_weights(
                    full_covariance[:-1, :-1], RIDGE_RATIOS[index]
                )
                weights = np.zeros(features)
                weights[:-1] = (1.0 - anchor) * reference_weights
                weights[-1] = anchor
            elif kind == "anchor_onehot":
                weights = np.zeros(features)
                weights[index] = 1.0 - anchor
                weights[-1] = anchor
            elif kind == "anchor_uniform":
                weights = np.zeros(features)
                weights[:-1] = (1.0 - anchor) / (features - 1)
                weights[-1] = anchor
            elif kind == "raw_noop":
                weights = np.zeros(features)
                weights[-1] = 1.0
            elif kind == "ridge":
                weights = _ridge_weights(full_covariance, RIDGE_RATIOS[index])
            elif kind == "one_hot":
                weights = np.zeros(features)
                weights[index] = 1.0
            else:
                weights = np.full(features, 1.0 / features)
            final_weights[lead, variable] = weights
            final_bias[lead, variable] = -float(weights @ full_mean)

    full_feature_map = train_feature_map + val_feature_map
    full_mean_map = (
        np.einsum("lvxym,lvm->lvxy", full_feature_map, final_weights, optimize=True)
        / float(train["days"] + validation["days"])
        + final_bias[..., None, None]
    )
    spatial_bias = spatial_shrink[..., None, None] * full_mean_map
    return {
        "weights": final_weights,
        "bias": final_bias,
        "spatial_bias": spatial_bias,
        "spatial_shrink": spatial_shrink,
        "selected_kind": selected_kind,
        "selected_index": selected_index,
        "selected_anchor": selected_anchor,
        "reference_count": np.asarray(reference_count),
    }


def fit_constrained_combination(
    train_statistics: dict[str, np.ndarray],
    validation_statistics: dict[str, np.ndarray],
    latitude: np.ndarray,
    minimum_backbone_weight: float = 0.5,
    backbone_index: int = 0,
) -> dict[str, np.ndarray]:
    """Public entry point of Mode B fitting (same computation as ``_fit_backbone``).

    The statistics dictionaries hold error moments of the R references and the
    backbone over the training / validation days (see
    scripts/spheredyn/run_main_spherettc_v2.py::_fit for how they are accumulated
    from 2017 forecasts).  Returns ``weights`` (L, V, R + 1; backbone last),
    ``bias`` (L, V), ``spatial_bias`` (L, V, H, W) and the selected candidates.
    """
    return _fit_backbone(
        backbone_index,
        train_statistics,
        validation_statistics,
        latitude,
        minimum_backbone_weight,
    )


def apply_constrained_combination(
    backbone: torch.Tensor | np.ndarray,
    references: Sequence[torch.Tensor | np.ndarray],
    fitted: dict[str, np.ndarray],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Apply frozen Mode B weights: (B, L, V, H, W) backbone + R references -> calibrated.

    Same arithmetic, in the same order and precision (float64), as the main
    SphereDyn run in scripts/spheredyn/run_main_spherettc_v2.py.
    ``references`` must follow the order of :data:`REFERENCES`.
    """
    device = torch.device(device)
    weights = torch.as_tensor(fitted["weights"], dtype=torch.float64, device=device)
    bias = torch.as_tensor(fitted["bias"], dtype=torch.float64, device=device)
    spatial_bias = torch.as_tensor(fitted["spatial_bias"], dtype=torch.float64, device=device)
    raw = torch.as_tensor(backbone, dtype=torch.float64, device=device)
    corrected = (
        bias[None, :, :, None, None]
        - spatial_bias[None]
        + raw * weights[None, :, :, -1, None, None]
    )
    for reference_index, prediction in enumerate(references):
        corrected = corrected + torch.as_tensor(
            prediction, dtype=torch.float64, device=device
        ) * weights[None, :, :, reference_index, None, None]
    return corrected
