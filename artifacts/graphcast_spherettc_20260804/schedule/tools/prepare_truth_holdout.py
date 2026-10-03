#!/usr/bin/env python
"""Materialize exact 1.5-degree 2020--2021-01-10 WeatherBench2 daily truth."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

import gcsfs
import numpy as np
import xarray as xr
import zarr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URI = (
    "weatherbench2/datasets/era5_daily/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"
)
PRESSURE = (
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
)
SURFACE = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
)
LEVELS = np.asarray([10, 50, 100, 200, 300, 500, 700, 850, 925, 1000], dtype=np.int32)
_thread = threading.local()


def dataset() -> xr.Dataset:
    if not hasattr(_thread, "dataset"):
        fs = gcsfs.GCSFileSystem(token="anon")
        _thread.dataset = xr.open_zarr(
            fs.get_mapper(SOURCE_URI), consolidated=True, chunks=None
        )
    return _thread.dataset


def shard_paths(root: Path, day: np.datetime64) -> tuple[Path, Path]:
    ymd = np.datetime_as_string(day, unit="D").replace("-", "")
    return (
        root / "pressure_level_1.5" / f"era5_pressure_full_1.5deg_{ymd}.zarr",
        root / "single_level_1.5" / f"era5_single_full_1.5deg_{ymd}.zarr",
    )


def valid_shards(pressure_path: Path, surface_path: Path) -> bool:
    try:
        pressure = zarr.open_group(str(pressure_path), mode="r")
        surface = zarr.open_group(str(surface_path), mode="r")
        return (
            np.array_equal(pressure["level"][:], LEVELS)
            and all(pressure[name].shape == (10, 121, 240) for name in PRESSURE)
            and all(surface[name].shape == (121, 240) for name in SURFACE)
        )
    except Exception:
        return False


def write_day(root: Path, day: np.datetime64) -> str:
    pressure_path, surface_path = shard_paths(root, day)
    if valid_shards(pressure_path, surface_path):
        return "reused"
    source = (
        dataset()
        .sel(time=day, level=LEVELS)
        .isel(latitude=slice(0, None, 6), longitude=slice(0, None, 6))
    )
    latitude = np.asarray(source.latitude.values, dtype=np.float32)
    longitude = np.asarray(source.longitude.values, dtype=np.float32)
    pressure_values = {
        name: np.asarray(source[name].values, dtype=np.float32) for name in PRESSURE
    }
    surface_values = {
        name: np.asarray(source[name].values, dtype=np.float32) for name in SURFACE
    }
    if not all(np.isfinite(value).all() for value in (*pressure_values.values(), *surface_values.values())):
        raise RuntimeError(f"non-finite truth on {day}")
    pressure_path.parent.mkdir(parents=True, exist_ok=True)
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    pressure = zarr.open_group(str(pressure_path), mode="w")
    pressure.create_dataset("level", data=LEVELS, overwrite=True)
    pressure.create_dataset("latitude", data=latitude, overwrite=True)
    pressure.create_dataset("longitude", data=longitude, overwrite=True)
    for name, value in pressure_values.items():
        pressure.create_dataset(name, data=value, chunks=(1, 121, 240), overwrite=True)
    surface = zarr.open_group(str(surface_path), mode="w")
    surface.create_dataset("latitude", data=latitude, overwrite=True)
    surface.create_dataset("longitude", data=longitude, overwrite=True)
    for name, value in surface_values.items():
        surface.create_dataset(name, data=value, chunks=(121, 240), overwrite=True)
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/S2S")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end-inclusive", default="2021-01-10")
    args = parser.parse_args()
    days = np.arange(
        np.datetime64(args.start, "D"),
        np.datetime64(args.end_inclusive, "D") + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    completed = 0
    counts = {"written": 0, "reused": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(write_day, args.root, day): day for day in days}
        for future in as_completed(futures):
            result = future.result()
            counts[result] += 1
            completed += 1
            if completed % 16 == 0 or completed == len(days):
                print(f"truth {completed}/{len(days)} {counts}", flush=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_uri": f"gs://{SOURCE_URI}",
        "start": args.start,
        "end_inclusive": args.end_inclusive,
        "days": len(days),
        "levels": LEVELS.tolist(),
        "pressure_variables": list(PRESSURE),
        "surface_variables": list(SURFACE),
        "subsample": "exact every-sixth 0.25-degree coordinate -> 1.5 degree",
        "counts": counts,
    }
    (args.root / "HOLDOUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("TRUTH_HOLDOUT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
