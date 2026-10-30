"""Public ERA5 preparation and the daily-field contract used by experiments."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import numpy as np
import xarray as xr

PRESSURE_LEVELS = (50, 100, 200, 300, 500, 700, 850, 925, 1000)
PRESSURE_NAMES = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}
SURFACE_NAMES = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "mslp": "mean_sea_level_pressure",
}
COMMON_VARIABLES = tuple(
    f"{prefix}{level}" for prefix in PRESSURE_NAMES for level in PRESSURE_LEVELS
) + tuple(SURFACE_NAMES)
SOURCE_URIS = {
    "main_daily": (
        "gs://weatherbench2/datasets/era5_daily/"
        "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"
    ),
    "conservative_6h": (
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    ),
}
LATITUDE = np.linspace(90.0, -90.0, 121)
LONGITUDE = np.arange(240) * 1.5


def canonical_variables(include_10hpa: bool = False) -> tuple[str, ...]:
    """Return the 49 evaluation fields, or the 54-field local-model contract."""
    if not include_10hpa:
        return COMMON_VARIABLES
    return tuple(
        f"{prefix}{level}hpa" if prefix in ("u", "v") and level == 10
        else f"{prefix}{level}"
        for prefix in PRESSURE_NAMES for level in (10, *PRESSURE_LEVELS)
    ) + tuple(SURFACE_NAMES)


def variable_source(variable: str) -> tuple[str, int | None]:
    """Map an unambiguous short name to the WeatherBench ERA5 variable."""
    if variable in SURFACE_NAMES:
        return SURFACE_NAMES[variable], None
    match = re.fullmatch(r"([zqtuv])(\d+)(?:hpa)?", variable)
    if match is None or variable not in canonical_variables(True):
        raise ValueError(f"Unknown field: {variable}")
    return PRESSURE_NAMES[match.group(1)], int(match.group(2))


def daily_dates(start: str, end: str) -> np.ndarray:
    """Inclusive UTC calendar-date range."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must not be after end")
    return np.arange(
        np.datetime64(first, "D"), np.datetime64(last, "D") + 1,
        dtype="datetime64[D]",
    )


def calendar_day_of_year(times: np.ndarray) -> np.ndarray:
    """Map month/day to 1..366 on the fixed leap-year 2000 calendar."""
    days = np.asarray(times).astype("datetime64[D]")
    month_day = np.datetime_as_string(days, unit="D")
    reference = np.asarray(["2000" + str(value)[4:] for value in month_day.flat],
                           dtype="datetime64[D]").reshape(days.shape)
    return ((reference - np.datetime64("2000-01-01", "D"))
            .astype("timedelta64[D]").astype(np.int16) + 1)


def normalize_grid(dataset: xr.Dataset) -> xr.Dataset:
    """Normalize coordinate names and ordering without changing values."""
    rename = {name: replacement for name, replacement in (
        ("latitude", "lat"), ("longitude", "lon"), ("pressure_level", "level")
    ) if name in dataset.dims}
    result = dataset.rename(rename)
    for name in ("time", "lat", "lon"):
        if name not in result.coords:
            raise ValueError(f"ERA5 source has no {name} coordinate")
    result = result.assign_coords(lon=np.mod(result.lon.values, 360.0))
    if len(np.unique(result.lon.values)) != result.sizes["lon"]:
        raise ValueError("Longitude normalization produced duplicate cells")
    return result.sortby("lat", ascending=False).sortby("lon")


def open_public_era5(source: str = "main_daily", uri: str | None = None) -> xr.Dataset:
    """Open anonymous public GCS metadata; field data remain lazy."""
    import fsspec

    if source not in SOURCE_URIS:
        raise ValueError(f"Unknown ERA5 source: {source}")
    mapper = fsspec.get_mapper(uri or SOURCE_URIS[source], token="anon")
    return normalize_grid(xr.open_zarr(mapper, consolidated=True, chunks=None))


def extract_daily_fields(
    dataset: xr.Dataset, day: np.datetime64, variables: tuple[str, ...] = COMMON_VARIABLES,
    source: str = "main_daily",
) -> xr.Dataset:
    """Load exactly one day, preserving the source-specific averaging rule."""
    if source not in SOURCE_URIS:
        raise ValueError(f"Unknown ERA5 source: {source}")
    variables = tuple(variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be nonempty and unique")
    day = np.datetime64(day, "D")
    requested = day.astype("datetime64[ns]") + np.asarray(
        [0, 6, 12, 18] if source == "conservative_6h" else [0],
        dtype="timedelta64[h]",
    )
    available = np.asarray(dataset.time.values).astype("datetime64[ns]")
    if not np.all(np.isin(requested, available)):
        raise ValueError(f"{day}: missing required UTC samples {requested[~np.isin(requested, available)]}")
    selected = dataset.sel(time=requested)
    if not np.array_equal(np.asarray(selected.time.values).astype("datetime64[ns]"), requested):
        raise ValueError(f"{day}: duplicate or unordered UTC samples")
    if source == "main_daily":
        # The main archive used the exact 1.5-degree subgrid of the daily source.
        selected = selected.isel(lat=slice(None, None, 6), lon=slice(None, None, 6))
    if (selected.sizes["lat"] != 121 or selected.sizes["lon"] != 240
            or not np.allclose(selected.lat.values, LATITUDE, atol=1e-6)
            or not np.allclose(selected.lon.values, LONGITUDE, atol=1e-6)):
        raise ValueError("Source does not resolve to the canonical 121 x 240 grid")
    arrays = []
    for variable in variables:
        name, level = variable_source(variable)
        if name not in selected:
            raise KeyError(f"ERA5 source is missing {name}")
        field = selected[name]
        if level is not None:
            field = field.sel(level=level)
        values = np.asarray(field.transpose("time", "lat", "lon").values, dtype=np.float32)
        if values.shape != (len(requested), 121, 240) or not np.isfinite(values).all():
            raise ValueError(f"{day}: nonfinite or incomplete field {variable}")
        arrays.append(values.mean(axis=0, dtype=np.float64).astype(np.float32))
    result = xr.Dataset(
        {"truth": (("time", "variable", "lat", "lon"), np.stack(arrays)[None])},
        coords={"time": [day.astype("datetime64[ns]")], "variable": list(variables),
                "lat": LATITUDE, "lon": LONGITUDE},
        attrs={"source_kind": source,
               "field_statistic": "UTC daily mean",
               "timestamp_definition": "target-day 00 UTC; not verification availability"},
    )
    return result


def write_daily_store(
    output: str | Path, start: str, end: str, source: str = "main_daily",
    variables: tuple[str, ...] = COMMON_VARIABLES, resume: bool = False,
) -> None:
    """Materialize one day at a time, with explicit source provenance."""
    import zarr
    from numcodecs import Blosc

    output = Path(output)
    dates = daily_dates(start, end)
    if output.exists() and not resume:
        raise FileExistsError(f"{output} exists; choose another path or use --resume")
    if output.exists():
        previous = xr.open_zarr(output, consolidated=True, chunks=None)
        try:
            if previous.attrs.get("source_kind") != source:
                raise ValueError("Cannot resume a different ERA5 source")
            if tuple(previous.variable.values.astype(str)) != tuple(variables):
                raise ValueError("Cannot resume a store with different variables")
            stored = np.asarray(previous.time.values).astype("datetime64[D]")
            if len(stored) > len(dates) or not np.array_equal(stored, dates[:len(stored)]):
                raise ValueError("Existing dates must be an exact prefix of the requested interval")
        finally:
            previous.close()
        dates = dates[len(stored):]
    if not len(dates):
        print(f"Complete: {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = open_public_era5(source)
    try:
        for index, day in enumerate(dates):
            daily = extract_daily_fields(dataset, day, tuple(variables), source)
            daily.attrs["source_uri"] = SOURCE_URIS[source]
            if not output.exists():
                encoding = {"truth": {
                    "chunks": (1, len(variables), 121, 240),
                    "compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
                }}
                daily.to_zarr(output, mode="w", consolidated=False, encoding=encoding)
            else:
                daily.to_zarr(output, mode="a", append_dim="time", consolidated=False)
            # Keep resumable metadata in sync after each fully written day.
            zarr.consolidate_metadata(str(output))
            if index % 10 == 0 or index == len(dates) - 1:
                print(f"ERA5 {index + 1}/{len(dates)}: {day}", flush=True)
    finally:
        dataset.close()


def prepare_statistics(
    truth_path: str | Path, output_dir: str | Path,
    fit_start: str = "1979-01-01", fit_end: str = "2016-12-31",
) -> None:
    """Stream fit-only normalization and 366-day climatology to disk.

    The variance is a population variance over dates and grid cells; latitude
    weighting is reserved for evaluation metrics. Calendar-day sums are stored
    on disk so memory use does not grow with the number of fit years.
    """
    import json
    import tempfile
    import zarr
    from numcodecs import Blosc

    dates = daily_dates(fit_start, fit_end)
    if dates[0] < np.datetime64("1979-01-01") or dates[-1] > np.datetime64("2016-12-31"):
        raise ValueError("Fit statistics must be restricted to the 1979--2016 training split")
    output_dir = Path(output_dir)
    climatology_path = output_dir / "climatology.zarr"
    stats_path = output_dir / "stats.json"
    if climatology_path.exists() or stats_path.exists():
        raise FileExistsError("Statistics already exist; use a new output directory")
    dataset = xr.open_zarr(truth_path, consolidated=True, chunks=None)
    try:
        if "truth" not in dataset or dataset.truth.dims != ("time", "variable", "lat", "lon"):
            raise ValueError("truth.zarr must contain truth(time, variable, lat, lon)")
        stored = np.asarray(dataset.time.values).astype("datetime64[ns]")
        wanted = dates.astype("datetime64[ns]")
        if (len(np.unique(stored)) != len(stored)
                or np.any(np.diff(stored) <= np.timedelta64(0, "ns"))
                or not np.all(np.isin(wanted, stored))):
            raise ValueError("truth.zarr does not cover every requested fit date exactly once")
        indices = np.searchsorted(stored, wanted)
        if not np.array_equal(stored[indices], wanted):
            raise ValueError("truth.zarr time must be strictly chronological daily dates")
        day_indices = calendar_day_of_year(dates) - 1
        day_counts = np.bincount(day_indices, minlength=366)
        if np.any(day_counts == 0):
            raise ValueError("Fit interval must cover every calendar day, including February 29")
        variables = np.asarray(dataset.variable.values).astype(str)
        latitude, longitude = np.asarray(dataset.lat.values), np.asarray(dataset.lon.values)
        shape = (len(variables), len(latitude), len(longitude))
        output_dir.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        count = 0
        mean = np.zeros(len(variables), dtype=np.float64)
        m2 = np.zeros_like(mean)
        with tempfile.TemporaryDirectory(prefix=".climatology-", dir=output_dir) as temporary:
            temporary_group = zarr.open_group(temporary, mode="w")
            sums = temporary_group.create_dataset(
                "calendar_sum", shape=(366, *shape), chunks=(1, *shape),
                dtype="f8", fill_value=0, compressor=compressor,
            )
            for step, (index, calendar_index) in enumerate(zip(indices, day_indices)):
                values = np.asarray(dataset.truth.isel(time=int(index)).values, dtype=np.float64)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite or incomplete fit truth on {dates[step]}")
                n_batch = shape[1] * shape[2]
                batch_mean = values.mean(axis=(1, 2))
                batch_m2 = ((values - batch_mean[:, None, None]) ** 2).sum(axis=(1, 2))
                difference = batch_mean - mean
                total = count + n_batch
                mean += difference * (n_batch / total)
                m2 += batch_m2 + difference**2 * (count * n_batch / total)
                count = total
                sums[int(calendar_index)] = sums[int(calendar_index)] + values
                if step % 365 == 0 or step == len(indices) - 1:
                    print(f"Fit statistics {step + 1}/{len(indices)}", flush=True)
            std = np.sqrt(m2 / count)
            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("Every variable needs a finite positive fit-split standard deviation")
            group = zarr.open_group(str(climatology_path), mode="w")
            group.attrs.update({
                "fit_start": fit_start, "fit_end": fit_end,
                "definition": "month/day mean on leap-year-2000 calendar; leap years only on February 29",
                "source_kind": dataset.attrs.get("source_kind", "unspecified"),
                "source_uri": dataset.attrs.get("source_uri", "unspecified"),
            })
            for name, array in (("dayofyear", np.arange(1, 367, dtype=np.int16)),
                                ("variable", variables), ("lat", latitude), ("lon", longitude)):
                coordinate = group.create_dataset(name, data=array, fill_value=None)
                coordinate.attrs["_ARRAY_DIMENSIONS"] = [name]
            result = group.create_dataset(
                "climatology", shape=(366, *shape), chunks=(1, *shape),
                dtype="f4", fill_value=np.nan, compressor=compressor,
            )
            result.attrs["_ARRAY_DIMENSIONS"] = ["dayofyear", "variable", "lat", "lon"]
            for day_index in range(366):
                result[day_index] = sums[day_index] / day_counts[day_index]
            counts = group.create_dataset("sample_count", data=day_counts, fill_value=None)
            counts.attrs["_ARRAY_DIMENSIONS"] = ["dayofyear"]
            zarr.consolidate_metadata(str(climatology_path))
        stats = {
            "variables": variables.tolist(), "mean": mean.tolist(),
            "standard_deviation": std.tolist(), "fit_start": fit_start, "fit_end": fit_end,
            "sample_count_per_variable": count,
            "variance_definition": "population variance over training dates and grid cells; uniform grid weights",
            "source_kind": dataset.attrs.get("source_kind", "unspecified"),
            "source_uri": dataset.attrs.get("source_uri", "unspecified"),
        }
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {stats_path} and {climatology_path}", flush=True)
    finally:
        dataset.close()
