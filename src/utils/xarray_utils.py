from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


PRESSURE_LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PRESSURE_LEVELS_10 = [100, 150, 200, 250, 300, 500, 700, 850, 925, 1000]
S2S_PRESSURE_LEVELS_10 = [10, 50, 100, 200, 300, 500, 700, 850, 925, 1000]
PRESSURE_VARIABLE_PREFIXES = {
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
}
SINGLE_LEVEL_VARIABLES = {
    "t2m": {"name": "2m_temperature", "level": None},
    "u10": {"name": "10m_u_component_of_wind", "level": None},
    "v10": {"name": "10m_v_component_of_wind", "level": None},
    "mslp": {"name": "mean_sea_level_pressure", "level": None},
    "tp": {"name": "total_precipitation", "level": None},
}


def expanded_medium_range_variables() -> list[str]:
    return [
        f"{prefix}{level}"
        for prefix in ["z", "t", "q", "u", "v"]
        for level in PRESSURE_LEVELS_10
    ] + list(SINGLE_LEVEL_VARIABLES)


def pnp_70_variables() -> list[str]:
    return [
        f"{prefix}{level}"
        for prefix in ["z", "q", "t", "u", "v"]
        for level in PRESSURE_LEVELS_13
    ] + ["u10", "v10", "t2m", "mslp", "tp"]


def pnp_69_variables() -> list[str]:
    return [
        f"{prefix}{level}"
        for prefix in ["z", "q", "t", "u", "v"]
        for level in PRESSURE_LEVELS_13
    ] + ["u10", "v10", "t2m", "mslp"]


CANONICAL_VARIABLES: Dict[str, Dict[str, object]] = {
    **{
        f"{prefix}{level}": {"name": source_name, "level": level}
        for prefix, source_name in PRESSURE_VARIABLE_PREFIXES.items()
        for level in PRESSURE_LEVELS_13
    },
    **SINGLE_LEVEL_VARIABLES,
    # Surface u10/v10 retain their established meaning (10 metre wind).  The
    # explicit hPa suffix avoids a collision with pressure-level wind at 10 hPa.
    "z10": {"name": "geopotential", "level": 10},
    "q10": {"name": "specific_humidity", "level": 10},
    "t10": {"name": "temperature", "level": 10},
    "u10hpa": {"name": "u_component_of_wind", "level": 10},
    "v10hpa": {"name": "v_component_of_wind", "level": 10},
}

NAME_ALIASES = {
    "geopotential": ["geopotential", "z", "z500"],
    "temperature": ["temperature", "t"],
    "specific_humidity": ["specific_humidity", "q"],
    "u_component_of_wind": ["u_component_of_wind", "u"],
    "v_component_of_wind": ["v_component_of_wind", "v"],
    "2m_temperature": ["2m_temperature", "t2m"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10"],
    "mean_sea_level_pressure": ["mean_sea_level_pressure", "msl", "mslp"],
    "total_precipitation": ["total_precipitation_6hr", "total_precipitation", "tp"],
}


def open_zarr_lazy(uri: str):
    import xarray as xr

    storage_options = {"token": "anon"} if uri.startswith("gs://") else None
    try:
        return xr.open_zarr(uri, consolidated=True, chunks={}, storage_options=storage_options)
    except AssertionError:
        ds = xr.open_zarr(
            uri,
            consolidated=True,
            chunks={},
            decode_times=False,
            storage_options=storage_options,
        )
        return decode_numeric_forecast_coords(ds)


def decode_numeric_forecast_coords(ds):
    import pandas as pd

    if "time" in ds.coords and not np.issubdtype(ds["time"].dtype, np.datetime64):
        units = ds["time"].attrs.get("units", "")
        if " since " in units:
            unit, origin = units.split(" since ", 1)
            unit = unit.strip().lower()
            origin = origin.strip()
            unit_map = {
                "hour": "h",
                "hours": "h",
                "day": "D",
                "days": "D",
            }
            if unit in unit_map:
                decoded = pd.Timestamp(origin) + pd.to_timedelta(ds["time"].values, unit=unit_map[unit])
                ds = ds.assign_coords(time=decoded.to_numpy(dtype="datetime64[ns]"))
    if "prediction_timedelta" in ds.coords and not np.issubdtype(ds["prediction_timedelta"].dtype, np.timedelta64):
        units = ds["prediction_timedelta"].attrs.get("units", "hours")
        if units in {"hour", "hours"}:
            values = np.asarray(ds["prediction_timedelta"].values).astype("timedelta64[h]")
            ds = ds.assign_coords(prediction_timedelta=values)
    return ds


def coord_name(ds, candidates: Iterable[str]) -> str | None:
    names = set(ds.coords) | set(ds.dims)
    for name in candidates:
        if name in names:
            return name
    return None


def normalize_lat_lon(ds):
    rename = {}
    lat = coord_name(ds, ["latitude", "lat"])
    lon = coord_name(ds, ["longitude", "lon"])
    if lat and lat != "lat":
        rename[lat] = "lat"
    if lon and lon != "lon":
        rename[lon] = "lon"
    if rename:
        ds = ds.rename(rename)
    flipped = False
    if "lat" in ds.coords and len(ds["lat"]) > 1 and float(ds["lat"][0]) < float(ds["lat"][-1]):
        ds = ds.sortby("lat", ascending=False)
        flipped = True
    # Some WeatherBench2 archives differ from ERA5 by ~1e-14 in float64
    # coordinates. Round before xarray alignment so targets do not become NaN.
    rounded = {}
    if "lat" in ds.coords:
        rounded["lat"] = np.round(np.asarray(ds["lat"].values, dtype=np.float64), 6)
    if "lon" in ds.coords:
        rounded["lon"] = np.round(np.asarray(ds["lon"].values, dtype=np.float64), 6)
    if rounded:
        ds = ds.assign_coords(**rounded)
    return ds, flipped


def time_dim(ds) -> str:
    name = coord_name(ds, ["init_time", "time"])
    if not name:
        raise KeyError("No init time coordinate found; expected init_time or time.")
    return name


def lead_dim(ds) -> str:
    name = coord_name(ds, ["prediction_timedelta", "lead_time", "lead_time_hours"])
    if not name:
        raise KeyError("No lead coordinate found; expected prediction_timedelta/lead_time.")
    return name


def level_dim(ds) -> str | None:
    return coord_name(ds, ["level", "pressure_level", "isobaricInhPa"])


def available_canonical_variables(ds, requested: Iterable[str] | None = None) -> Tuple[List[str], List[str]]:
    requested = list(requested or CANONICAL_VARIABLES)
    available, missing = [], []
    data_vars = set(ds.data_vars)
    lvl = level_dim(ds)
    levels = set()
    if lvl and lvl in ds.coords:
        levels = {int(x) for x in np.asarray(ds[lvl].values).astype(int).tolist()}
    for canonical in requested:
        spec = CANONICAL_VARIABLES[canonical]
        found_name = any(alias in data_vars for alias in NAME_ALIASES[str(spec["name"])])
        found_level = spec["level"] is None or not lvl or int(spec["level"]) in levels
        if found_name and found_level:
            available.append(canonical)
        else:
            missing.append(canonical)
    return available, missing


def select_canonical_dataarray(ds, canonical: str):
    spec = CANONICAL_VARIABLES[canonical]
    aliases = NAME_ALIASES[str(spec["name"])]
    source_name = next((name for name in aliases if name in ds.data_vars), None)
    if not source_name:
        raise KeyError(f"{canonical} source variable not found in dataset.")
    da = ds[source_name]
    lvl = level_dim(ds)
    if spec["level"] is not None and lvl and lvl in da.dims:
        da = da.sel({lvl: int(spec["level"])})
    return da
