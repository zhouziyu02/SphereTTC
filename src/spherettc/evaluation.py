"""Streaming latitude-weighted RMSE and pooled anomaly correlation."""

from __future__ import annotations

import numpy as np


def calendar_day_numbers(dates) -> np.ndarray:
    """Map month/day to 1--366 using the leap-year 2000 calendar."""
    dates = np.asarray(dates, dtype="datetime64[D]")
    result = np.empty(dates.size, dtype=np.int16)
    for index, value in enumerate(dates.reshape(-1)):
        if np.isnat(value):
            raise ValueError("Dates cannot contain NaT.")
        month_day = np.datetime_as_string(value, unit="D")[5:]
        day = np.datetime64(f"2000-{month_day}", "D")
        result[index] = int((day - np.datetime64("2000-01-01", "D")) / np.timedelta64(1, "D")) + 1
    return result.reshape(dates.shape)


def validate_grid(lat, lon) -> None:
    """Require a complete, equally spaced equiangular global grid."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or min(lat.size, lon.size) < 2:
        raise ValueError("Latitude and longitude must be one-dimensional grid coordinates.")
    if not np.allclose(lat, np.linspace(90, -90, lat.size), atol=1e-5, rtol=0):
        raise ValueError("Latitude must be a complete, descending equiangular 90 to -90 grid.")
    if not np.allclose(lon, np.arange(lon.size) * (360 / lon.size), atol=1e-5, rtol=0):
        raise ValueError("Longitude must be a complete, ascending equiangular [0, 360) grid.")


def require_same_coordinates(reference, other, names=("variable", "lat", "lon")) -> None:
    """Reject misaligned fields instead of silently interpolating or intersecting."""
    for name in names:
        if name not in reference.coords or name not in other.coords:
            raise ValueError(f"Missing coordinate: {name}")
        if not np.array_equal(reference[name].values, other[name].values):
            raise ValueError(f"Coordinates differ for {name}; prepare matching fields first.")


class PooledMetrics:
    """Accumulate sufficient statistics without retaining forecast fields.

    Each update has shape (lead, variable, latitude, longitude). RMSE pools
    squared errors before its square root. ACC pools cross-products and both
    anomaly energies before normalization; daily correlations are not averaged.
    """

    def __init__(self, n_leads: int, n_variables: int, lat):
        weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64))).clip(0)
        self.weights = weights / weights.sum()
        self.shape = (int(n_leads), int(n_variables))
        self.count = 0
        self.sums = {
            method: {key: np.zeros(self.shape, np.float64) for key in ("mse", "cross", "prediction_energy", "target_energy")}
            for method in ("raw", "spherettc")
        }

    def _spatial_mean(self, field: np.ndarray) -> np.ndarray:
        return np.sum(field.mean(axis=-1) * self.weights, axis=-1)

    def update(self, raw, calibrated, target, climatology) -> None:
        raw, calibrated, target, climatology = (
            np.asarray(item, dtype=np.float64) for item in (raw, calibrated, target, climatology)
        )
        if raw.ndim != 4 or raw.shape[:2] != self.shape or raw.shape[-2] != self.weights.size:
            raise ValueError("Forecast must have shape (lead, variable, latitude, longitude).")
        if not all(item.shape == raw.shape for item in (calibrated, target, climatology)):
            raise ValueError("Forecast, target, and climatology shapes must match exactly.")
        if not all(np.isfinite(item).all() for item in (raw, calibrated, target, climatology)):
            raise ValueError("Non-finite fields are not accepted in the paired evaluation.")
        target_anomaly = target - climatology
        for method, prediction in (("raw", raw), ("spherettc", calibrated)):
            sums = self.sums[method]
            prediction_anomaly = prediction - climatology
            sums["mse"] += self._spatial_mean((prediction - target) ** 2)
            sums["cross"] += self._spatial_mean(prediction_anomaly * target_anomaly)
            sums["prediction_energy"] += self._spatial_mean(prediction_anomaly ** 2)
            sums["target_energy"] += self._spatial_mean(target_anomaly ** 2)
        self.count += 1

    def scores(self) -> dict[str, dict[str, np.ndarray]]:
        if not self.count:
            raise ValueError("Cannot score an empty evaluation period.")
        scores = {}
        for method, sums in self.sums.items():
            denominator = np.sqrt(sums["prediction_energy"] * sums["target_energy"])
            if np.any(denominator <= 0):
                raise ValueError("ACC is undefined for a zero-energy anomaly series.")
            scores[method] = {
                "rmse": np.sqrt(sums["mse"] / self.count),
                "acc": sums["cross"] / denominator,
            }
        return scores


def macro_scores(scores, standard_deviation) -> dict:
    """Average normalized per-variable RMSE, then average lead times equally."""
    sigma = np.asarray(standard_deviation, dtype=np.float64)
    if sigma.ndim != 1 or np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("Standard deviations must be finite and positive.")
    result = {}
    for method in ("raw", "spherettc"):
        rmse, acc = (np.asarray(scores[method][key], np.float64) for key in ("rmse", "acc"))
        if rmse.ndim != 2 or rmse.shape != acc.shape or rmse.shape[1] != sigma.size:
            raise ValueError("Scores must have matching (lead, variable) dimensions.")
        nrmse = (rmse / sigma).mean(axis=1)
        acc_mean = acc.mean(axis=1)
        result[method] = {
            "nrmse_by_lead": nrmse.tolist(),
            "acc_by_lead": acc_mean.tolist(),
            "overall_nrmse": float(nrmse.mean()),
            "overall_acc": float(acc_mean.mean()),
        }
    return result


def relative_improvements(raw_nrmse: float, calibrated_nrmse: float, raw_acc: float, calibrated_acc: float) -> tuple:
    """Return percentage improvements; undefined relative changes are None."""
    nrmse = 100 * (1 - calibrated_nrmse / raw_nrmse) if raw_nrmse > 0 else None
    acc = 100 * (calibrated_acc / raw_acc - 1) if raw_acc > 0 else None
    return nrmse, acc
