#!/usr/bin/env python3
"""Download public GraphCast/Pangu daily-mean forecasts for a small replay.

These WeatherBench2 forecast archives are a public-data example, not substitutes
for the paper's original eleven-backbone checkpoint-generated main caches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

from spherettc.data import daily_dates, normalize_grid, variable_source

SOURCES = {
    "graphcast": (
        "gs://weatherbench2/datasets/graphcast/2018/"
        "date_range_2017-11-16_2019-02-01_12_hours-"
        "240x121_equiangular_with_poles_conservative.zarr"
    ),
    "pangu": (
        "gs://weatherbench2/datasets/pangu/"
        "2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr"
    ),
}


def extract_forecast(dataset, init_time, leads, variables):
    """Average the four subdaily states of each requested target day."""
    init_time = np.datetime64(init_time, "ns")
    if init_time not in dataset.time.values:
        raise ValueError(f"Source has no 00 UTC forecast for {init_time}.")
    steps = dataset.prediction_timedelta.values.astype("timedelta64[h]")
    arrays = []
    for lead in leads:
        needed = np.asarray([lead + offset for offset in (0, 6, 12, 18)], dtype="timedelta64[h]")
        if not np.isin(needed, steps).all():
            raise ValueError(f"Source lacks all four daily samples for lead {lead} h.")
        fields = []
        for variable in variables:
            name, level = variable_source(variable)
            value = dataset[name].sel(time=init_time, prediction_timedelta=needed)
            if level is not None:
                value = value.sel(level=level)
            field = value.mean("prediction_timedelta", skipna=False).transpose("lat", "lon")
            fields.append(np.asarray(field.values, dtype=np.float32))
        arrays.append(np.stack(fields))
    prediction = np.stack(arrays)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite forecast at {init_time}.")
    return xr.Dataset(
        {"prediction": (("init_time", "lead_time", "variable", "lat", "lon"), prediction[None])},
        coords={"init_time": [init_time], "lead_time": leads, "variable": variables,
                "lat": dataset.lat.values, "lon": dataset.lon.values},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SOURCES, required=True)
    parser.add_argument("--start", required=True, help="First initialization date, inclusive.")
    parser.add_argument("--end", required=True, help="Last initialization date, inclusive.")
    parser.add_argument("--variables", nargs="+", default=["z500", "t850", "t2m", "mslp", "u10"])
    parser.add_argument("--leads", nargs="+", type=int, default=[24, 72, 120, 168])
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without network access.")
    args = parser.parse_args()
    dates = daily_dates(args.start, args.end)
    if len(set(args.variables)) != len(args.variables) or not args.variables:
        parser.error("Variables must be nonempty and unique.")
    for variable in args.variables:
        variable_source(variable)
    if args.leads != sorted(set(args.leads)) or any(x <= 0 or x % 24 or x + 18 > 240 for x in args.leads):
        parser.error("Leads must be increasing whole days; each lead + 18 h must be <= 240 h.")
    plan = {"source_uri": SOURCES[args.model], "model": args.model,
            "start": args.start, "end": args.end, "variables": args.variables,
            "lead_hours": args.leads, "daily_sample_hours": [0, 6, 12, 18],
            "protocol": "public_archive_example", "target_grid": [121, 240],
            "uncompressed_output_bytes": int(len(dates) * len(args.leads) * len(args.variables) * 121 * 240 * 4)}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists. Use a new output or --resume.")
        with xr.open_zarr(output, consolidated=True) as old:
            completed = old.sizes["init_time"]
            for key in ("source_uri", "protocol"):
                if old.attrs.get(key) != plan[key]:
                    raise ValueError(f"Resume metadata differs for {key}.")
            if not np.array_equal(old.init_time.values.astype("datetime64[D]"), dates[:completed]):
                raise ValueError("Existing dates must be an exact prefix of the requested range.")
            if old.variable.values.tolist() != args.variables or old.lead_time.values.tolist() != args.leads:
                raise ValueError("Resume variables or leads differ.")
    mapper = fsspec.get_mapper(SOURCES[args.model], token="anon")
    with xr.open_zarr(mapper, consolidated=True, chunks=None) as remote:
        source = normalize_grid(remote)
        if source.sizes["lat"] != 121 or source.sizes["lon"] != 240:
            raise ValueError("Unexpected public forecast grid.")
        for i, day in enumerate(dates[completed:], start=completed):
            item = extract_forecast(source, day, args.leads, args.variables)
            item.attrs.update({key: value for key, value in plan.items() if key != "uncompressed_output_bytes"})
            if i == 0:
                item.to_zarr(output, mode="w", consolidated=True,
                             encoding={"prediction": {"chunks": (1, 1, 1, 121, 240)}})
            else:
                item.to_zarr(output, mode="a", append_dim="init_time", consolidated=True)
            print(f"Saved {day} ({i + 1}/{len(dates)})", flush=True)
    output.with_suffix(".manifest.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
