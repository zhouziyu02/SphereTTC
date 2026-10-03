from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr

from src.experiments.protocol import PRESSURE_LEVELS, PROTOCOL_VARIABLES


PRESSURE_SOURCES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
SURFACE_SOURCES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]
LEAD_HOURS = np.asarray([24, 72, 120, 168, 240], dtype=np.int32)


def _as_day(value) -> np.datetime64:
    return np.datetime64(value, "D")


def shard_paths(root: str | Path, day) -> tuple[Path, Path]:
    root = Path(root)
    ymd = np.datetime_as_string(_as_day(day), unit="D").replace("-", "")
    return (
        root / "pressure_level_1.5" / f"era5_pressure_full_1.5deg_{ymd}.zarr",
        root / "single_level_1.5" / f"era5_single_full_1.5deg_{ymd}.zarr",
    )


def load_day(root: str | Path, day, require_finite: bool = True) -> np.ndarray:
    """Load one canonical 54-variable day as (variable, lat, lon)."""
    pressure_path, surface_path = shard_paths(root, day)
    pressure = zarr.open_group(str(pressure_path), mode="r")
    surface = zarr.open_group(str(surface_path), mode="r")
    levels = np.asarray(pressure["level"][:], dtype=np.int32)
    if not np.array_equal(levels, np.asarray(PRESSURE_LEVELS, dtype=np.int32)):
        raise RuntimeError(f"pressure levels mismatch in {pressure_path}: {levels.tolist()}")

    arrays = []
    for source in PRESSURE_SOURCES:
        if source not in pressure:
            raise KeyError(f"{pressure_path} missing {source}")
        value = np.asarray(pressure[source][:], dtype=np.float32)
        if value.shape != (10, 121, 240):
            raise RuntimeError(f"{pressure_path}/{source} has shape {value.shape}")
        arrays.append(value)
    for source in SURFACE_SOURCES:
        if source not in surface:
            raise KeyError(f"{surface_path} missing {source}")
        value = np.asarray(surface[source][:], dtype=np.float32)
        if value.shape != (121, 240):
            raise RuntimeError(f"{surface_path}/{source} has shape {value.shape}")
        arrays.append(value[None, ...])
    result = np.concatenate(arrays, axis=0)
    if result.shape != (len(PROTOCOL_VARIABLES), 121, 240):
        raise RuntimeError(f"canonical day has unexpected shape {result.shape}")
    if require_finite and not np.isfinite(result).all():
        raise RuntimeError(f"non-finite values in S2S day {_as_day(day)}")
    return result


def load_coordinates(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    pressure_path, _ = shard_paths(root, "1979-01-01")
    if not pressure_path.is_dir():
        candidates = sorted(
            (root / "pressure_level_1.5").glob(
                "era5_pressure_full_1.5deg_*.zarr"
            )
        )
        if not candidates:
            raise RuntimeError(
                f"{root} has no canonical pressure-level daily shards"
            )
        pressure_path = candidates[0]
    group = zarr.open_group(str(pressure_path), mode="r")
    return np.asarray(group["latitude"][:], dtype=np.float32), np.asarray(group["longitude"][:], dtype=np.float32)


def load_targets(
    root: str | Path,
    init_times: Iterable[np.datetime64],
    lead_hours: Iterable[int],
    variables: Iterable[str] | None = None,
    workers: int = 8,
) -> np.ndarray:
    """Load truth for arbitrary init/lead pairs from the canonical daily shards."""
    init = np.asarray(list(init_times), dtype="datetime64[D]")
    leads = np.asarray(list(lead_hours), dtype=np.int32)
    if np.any(leads % 24):
        raise ValueError("daily S2S truth only supports lead times divisible by 24 hours")
    target_days = init[:, None] + (leads // 24)[None, :].astype("timedelta64[D]")
    unique = sorted({_as_day(day) for day in target_days.reshape(-1)})
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(unique) or 1)) as pool:
        loaded = dict(zip(unique, pool.map(lambda day: load_day(root, day), unique)))
    selected = list(variables or PROTOCOL_VARIABLES)
    indices = []
    for variable in selected:
        if variable not in PROTOCOL_VARIABLES:
            raise KeyError(f"unknown S2S truth variable: {variable}")
        indices.append(PROTOCOL_VARIABLES.index(variable))
    return np.stack(
        [np.stack([loaded[_as_day(day)][indices] for day in row]) for row in target_days]
    ).astype(np.float32, copy=False)


def daily_range(start: str, end_exclusive: str) -> np.ndarray:
    return np.arange(np.datetime64(start, "D"), np.datetime64(end_exclusive, "D"), np.timedelta64(1, "D"))


class S2SDailyBackend:
    """Direct random-access backend for local baseline training and evaluation."""

    def __init__(
        self,
        root: str | Path,
        stats_path: str | Path,
        io_workers: int = 8,
        contiguous_cache: str | Path | None = None,
    ):
        self.root = Path(root)
        self.variables = list(PROTOCOL_VARIABLES)
        self.lead_time = LEAD_HOURS.copy()
        self.fit_init_time = daily_range("1979-01-01", "2017-01-01")
        # The 240 h horizon from the final fit initialization (2016-12-31)
        # reaches 2017-01-10.  Start validation after a full-horizon embargo
        # so no validation initialization has appeared as a fit target.
        self.validation_init_time = daily_range("2017-01-11", "2018-01-01")
        self.train_init_time = np.concatenate([self.fit_init_time, self.validation_init_time])
        self.eval_init_time = daily_range("2018-01-01", "2020-01-01")
        self.lat, self.lon = load_coordinates(self.root)
        self.io_workers = max(1, int(io_workers))
        self.contiguous_cache = None
        self.contiguous_cache_start = None
        self.contiguous_cache_end = None
        if contiguous_cache is not None:
            cache_root = Path(contiguous_cache)
            manifest = json.loads(
                (cache_root / "MANIFEST.json").read_text(encoding="utf-8")
            )
            if manifest.get("prospective_2023_touched") is not False:
                raise RuntimeError("contiguous daily cache lacks the 2023 isolation guarantee")
            if manifest.get("variable_order") != self.variables:
                raise RuntimeError("contiguous daily cache variable order mismatch")
            if manifest.get("dtype") != "float32" or manifest.get("shape", [])[1:] != [54, 121, 240]:
                raise RuntimeError("contiguous daily cache shape/dtype mismatch")
            self.contiguous_cache_start = _as_day(manifest["start_date"])
            self.contiguous_cache_end = _as_day(manifest["end_date_inclusive"])
            if self.contiguous_cache_end >= np.datetime64("2023-01-01", "D"):
                raise RuntimeError("development cache must end before frozen 2023")
            self.contiguous_cache = np.load(
                cache_root / "daily.npy", mmap_mode="r", allow_pickle=False
            )
            if list(self.contiguous_cache.shape) != manifest["shape"]:
                raise RuntimeError("contiguous daily cache manifest shape mismatch")

        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        if stats.get("variables") != self.variables:
            raise RuntimeError("normalization statistics variable order does not match the daily protocol")
        if stats.get("fit_init_start") != "1979-01-01" or stats.get("fit_init_end") != "2016-12-31":
            raise RuntimeError("normalization statistics are not restricted to the 1979-2016 fit split")
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        if self.mean.shape != (54,) or self.std.shape != (54,):
            raise RuntimeError("normalization statistics must each have 54 entries")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or not (self.std > 0).all():
            raise RuntimeError("normalization statistics are non-finite or non-positive")

    def _load_unique_days(self, days: Iterable[np.datetime64]) -> dict[np.datetime64, np.ndarray]:
        unique = sorted({_as_day(day) for day in days})
        if self.contiguous_cache is not None:
            assert self.contiguous_cache_start is not None
            assert self.contiguous_cache_end is not None
            if unique and (
                unique[0] < self.contiguous_cache_start
                or unique[-1] > self.contiguous_cache_end
            ):
                raise RuntimeError(
                    "requested day is outside the frozen development cache: "
                    f"{unique[0]}..{unique[-1]}"
                )
            indices = [
                int((day - self.contiguous_cache_start) / np.timedelta64(1, "D"))
                for day in unique
            ]
            return {
                day: np.asarray(self.contiguous_cache[index])
                for day, index in zip(unique, indices, strict=True)
            }
        with ThreadPoolExecutor(max_workers=min(self.io_workers, len(unique) or 1)) as pool:
            values = list(pool.map(lambda day: load_day(self.root, day), unique))
        return dict(zip(unique, values))

    def load_batch(self, init_times: Iterable[np.datetime64]) -> tuple[np.ndarray, np.ndarray]:
        init = np.asarray(list(init_times), dtype="datetime64[D]")
        lead_days = (self.lead_time // 24).astype(np.int32)
        target_days = init[:, None] + lead_days[None, :].astype("timedelta64[D]")
        loaded = self._load_unique_days(np.concatenate([init, target_days.reshape(-1)]))
        inputs = np.stack([loaded[_as_day(day)] for day in init])
        targets = np.stack(
            [np.stack([loaded[_as_day(day)] for day in row]) for row in target_days]
        )
        return inputs, targets

    def load_batch_with_history(
        self,
        init_times: Iterable[np.datetime64],
        history_steps: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load chronological daily history plus the standard five targets."""
        if int(history_steps) < 1:
            raise ValueError("history_steps must be positive")
        init = np.asarray(list(init_times), dtype="datetime64[D]")
        offsets = np.arange(
            int(history_steps) - 1,
            -1,
            -1,
            dtype=np.int32,
        )
        history_days = init[:, None] - offsets[None].astype("timedelta64[D]")
        lead_days = (self.lead_time // 24).astype(np.int32)
        target_days = init[:, None] + lead_days[None].astype("timedelta64[D]")
        loaded = self._load_unique_days(
            np.concatenate([history_days.reshape(-1), target_days.reshape(-1)])
        )
        history = np.stack(
            [
                np.stack([loaded[_as_day(day)] for day in row])
                for row in history_days
            ]
        )
        targets = np.stack(
            [
                np.stack([loaded[_as_day(day)] for day in row])
                for row in target_days
            ]
        )
        return history, targets

    def load_history_inputs(
        self,
        init_times: Iterable[np.datetime64],
        history_steps: int = 3,
    ) -> np.ndarray:
        """Load chronological inputs without future targets."""
        if int(history_steps) < 1:
            raise ValueError("history_steps must be positive")
        init = np.asarray(list(init_times), dtype="datetime64[D]")
        offsets = np.arange(
            int(history_steps) - 1,
            -1,
            -1,
            dtype=np.int32,
        )
        history_days = init[:, None] - offsets[None].astype("timedelta64[D]")
        loaded = self._load_unique_days(history_days.reshape(-1))
        return np.stack(
            [
                np.stack([loaded[_as_day(day)] for day in row])
                for row in history_days
            ]
        )

    def load_inputs(self, init_times: Iterable[np.datetime64]) -> np.ndarray:
        """Load only initialization fields for prediction-only inference."""
        init = np.asarray(list(init_times), dtype="datetime64[D]")
        loaded = self._load_unique_days(init)
        return np.stack([loaded[_as_day(day)] for day in init])

    def load_train_indices(self, indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        return self.load_batch(self.train_init_time[np.asarray(list(indices), dtype=np.int64)])

    def load_eval_indices(self, indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        return self.load_batch(self.eval_init_time[np.asarray(list(indices), dtype=np.int64)])
