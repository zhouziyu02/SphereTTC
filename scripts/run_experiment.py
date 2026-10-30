#!/usr/bin/env python3
"""Replay one frozen forecaster chronologically and evaluate Raw/SphereTTC."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import xarray as xr

from spherettc import SphereTTC, SphereTTCConfig
from spherettc.evaluation import (
    PooledMetrics,
    calendar_day_numbers,
    macro_scores,
    relative_improvements,
    require_same_coordinates,
    validate_grid,
)


COMMON_VARIABLES = [
    f"{prefix}{level}"
    for prefix in ("z", "q", "t", "u", "v")
    for level in (50, 100, 200, 300, 500, 700, 850, 925, 1000)
] + ["u10", "v10", "t2m", "mslp"]


def open_field(path: str, candidates: tuple[str, ...], dimensions: tuple[str, ...]) -> xr.DataArray:
    dataset = xr.open_dataset(path) if Path(path).suffix in (".nc", ".netcdf") else xr.open_zarr(path, consolidated=None)
    aliases = {"latitude": "lat", "longitude": "lon"}
    dataset = dataset.rename({key: value for key, value in aliases.items() if key in dataset.dims})
    choices = [name for name in candidates if name in dataset.data_vars]
    if not choices:
        choices = [name for name, value in dataset.data_vars.items() if set(value.dims) == set(dimensions)]
    if len(choices) != 1:
        raise ValueError(f"{path}: expected one field named {candidates}; found {choices}.")
    field = dataset[choices[0]]
    if set(field.dims) != set(dimensions):
        raise ValueError(f"{path}: expected dimensions {dimensions}, got {field.dims}.")
    field = field.transpose(*dimensions)
    field.attrs = {**dataset.attrs, **field.attrs}
    if "valid_time" in dataset and dataset["valid_time"].dims == ("init_time", "lead_time"):
        field = field.assign_coords(valid_time=dataset["valid_time"])
    for name in dimensions:
        if name not in field.coords or field[name].ndim != 1:
            raise ValueError(f"{path}: coordinate {name} must be one-dimensional.")
        if np.unique(field[name].values).size != field.sizes[name]:
            raise ValueError(f"{path}: duplicate coordinate values in {name}.")
    validate_grid(field.lat.values, field.lon.values)
    return field


def forecast_field(path: str, variables: list[str]) -> tuple[xr.DataArray, np.ndarray]:
    field = open_field(path, ("prediction", "pred", "forecast"), ("init_time", "lead_time", "variable", "lat", "lon"))
    if not set(variables).issubset(set(map(str, field["variable"].values))):
        raise ValueError(f"{path}: does not contain all selected variables.")
    field = field.sel(variable=variables)
    values = field.lead_time.values
    if np.issubdtype(values.dtype, np.timedelta64):
        hours = values / np.timedelta64(1, "h")
    else:
        hours = np.asarray(values, dtype=np.float64)
    if not np.isfinite(hours).all() or not np.allclose(hours, np.round(hours)):
        raise ValueError("Lead times must be finite integer hours or timedeltas.")
    hours = hours.astype(np.int64)
    if np.any(hours <= 0) or np.any(hours % 24) or np.any(np.diff(hours) <= 0):
        raise ValueError("Daily fields require positive, increasing lead times divisible by 24 hours.")
    field = field.assign_coords(lead_time=hours)
    times = field.init_time.values.astype("datetime64[ns]")
    if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Forecast initialization times must be unique and strictly increasing.")
    if np.any(times != times.astype("datetime64[D]")):
        raise ValueError("The daily main protocol requires 00 UTC initialization timestamps.")
    if "valid_time" in field.coords:
        expected = times[:, None] + hours.astype("timedelta64[h]")[None, :]
        if not np.array_equal(field.valid_time.values.astype("datetime64[ns]"), expected):
            raise ValueError("Cache valid_time does not equal init_time plus lead_time.")
    return field, hours


def read_parameters(path: str) -> tuple[dict, int]:
    payload = json.loads(Path(path).read_text())
    parameters = payload.get("params", payload.get("best_params", payload))
    parameters = dict(parameters)
    memory_size = int(parameters.pop("memory_size", 64))
    unknown = set(parameters) - {item.name for item in fields(SphereTTCConfig)}
    if unknown:
        raise ValueError(f"Unknown calibrator parameters: {sorted(unknown)}")
    if memory_size <= 0:
        raise ValueError("memory_size must be positive.")
    return parameters, memory_size


def read_statistics(path: str, variables: list[str]) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    fit_end = payload.get("fit_end", payload.get("fit_init_end"))
    fit_end = np.datetime64(fit_end, "D") if fit_end is not None else np.datetime64("NaT", "D")
    if np.isnat(fit_end) or fit_end > np.datetime64("2016-12-31", "D"):
        raise ValueError("Normalization statistics must document a fit split ending no later than 2016-12-31.")
    std = payload.get("standard_deviation", payload.get("std"))
    if std is None or len(payload["variables"]) != len(std):
        raise ValueError("Statistics must contain equally sized variables and standard_deviation arrays.")
    lookup = dict(zip(payload["variables"], std, strict=True))
    if len(lookup) != len(std) or not set(variables).issubset(lookup):
        raise ValueError("Statistics contain duplicate names or omit selected variables.")
    values = np.asarray([lookup[name] for name in variables], np.float64)
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Normalization standard deviations must be finite and positive.")
    return values


def file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def climatology_metadata(field: xr.DataArray, path: str) -> dict:
    """Validate the fit split and fingerprint metadata without reading all fields."""
    first, last = field.attrs.get("fit_start"), field.attrs.get("fit_end")
    if first is None or last is None:
        raise ValueError("Climatology must document fit_start and fit_end; rebuild it with prepare_statistics.py.")
    first, last = np.datetime64(first, "D"), np.datetime64(last, "D")
    if np.isnat(first) or np.isnat(last) or first > last or last > np.datetime64("2016-12-31", "D"):
        raise ValueError("Climatology must be fitted exclusively on the pre-2017 training split.")
    digest = hashlib.sha256(json.dumps(field.attrs, sort_keys=True, default=str).encode())
    for coordinate in ("dayofyear", "variable", "lat", "lon"):
        digest.update(json.dumps(field[coordinate].values.tolist(), default=str).encode())
    metadata = Path(path) / ".zmetadata"
    if metadata.is_file():
        digest.update(metadata.read_bytes())
    return {
        "fit_start": str(first), "fit_end": str(last),
        "metadata_sha256": digest.hexdigest(),
        "fingerprint_scope": "attributes, coordinates, and consolidated Zarr metadata; excludes array chunks",
        "source_kind": field.attrs.get("source_kind", "unspecified"),
    }


def require_matching_sources(**sources) -> None:
    known = {name: source for name, source in sources.items() if source not in (None, "", "unspecified", "unknown")}
    if len(set(known.values())) > 1:
        raise ValueError(f"Truth, climatology, and statistics use different preprocessing sources: {known}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-cache", required=True)
    parser.add_argument("--truth", required=True, help="Daily ERA5 truth.zarr produced by download_era5.py.")
    parser.add_argument("--climatology", required=True, help="Training-split 366-calendar-day climatology.zarr.")
    parser.add_argument("--statistics", required=True, help="Training-split variable standard deviations JSON.")
    parser.add_argument("--params", required=True, help="Frozen model-specific parameter JSON.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", default="2018-01-01", help="First scored initialization, inclusive.")
    parser.add_argument("--end", default="2019-12-31", help="Last scored initialization, inclusive.")
    parser.add_argument("--warmup-cache", help="Optional cache with initializations before --start.")
    parser.add_argument("--require-warmup", action="store_true", help="Fail unless a full eligible memory exists at the first scored initialization.")
    parser.add_argument("--verification-delay-hours", type=float, default=0, help="0 reproduces archived target-date admission; 24 admits completed daily means.")
    parser.add_argument("--variables", nargs="+", default=COMMON_VARIABLES, help="Default: all 49 common main-experiment variables.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, help="New output directory; existing results are never overwritten.")
    args = parser.parse_args()
    variables = args.variables
    if len(set(variables)) != len(variables):
        parser.error("--variables contains duplicates.")
    if args.verification_delay_hours < 0 or not np.isfinite(args.verification_delay_hours):
        parser.error("--verification-delay-hours must be finite and nonnegative.")
    start, end = np.datetime64(args.start, "D"), np.datetime64(args.end, "D")
    if np.isnat(start) or np.isnat(end) or start > end:
        parser.error("--start and --end must define a nonempty period.")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory.")

    prediction, leads = forecast_field(args.forecast_cache, variables)
    truth = open_field(args.truth, ("truth", "target"), ("time", "variable", "lat", "lon")).sel(variable=variables)
    climatology = open_field(args.climatology, ("climatology",), ("dayofyear", "variable", "lat", "lon")).sel(variable=variables)
    require_same_coordinates(prediction, truth)
    require_same_coordinates(prediction, climatology)
    if not np.array_equal(climatology.dayofyear.values, np.arange(1, 367)):
        raise ValueError("Climatology must use calendar-day numbers 1--366 on a leap-year template.")
    climatology_info = climatology_metadata(climatology, args.climatology)
    statistics_info = json.loads(Path(args.statistics).read_text())
    source_kinds = {
        "truth": truth.attrs.get("source_kind", "unspecified"),
        "climatology": climatology_info["source_kind"],
        "statistics": statistics_info.get("source_kind", "unspecified"),
    }
    require_matching_sources(**source_kinds)
    sigma = read_statistics(args.statistics, variables)
    parameters, memory_size = read_parameters(args.params)
    config = SphereTTCConfig(**parameters)
    calibrator = SphereTTC(
        prediction.sizes["lat"], prediction.sizes["lon"], lead_hours=leads,
        config=config, memory_size=memory_size, device=args.device,
        verification_delay_hours=args.verification_delay_hours,
    )
    times = prediction.init_time.values.astype("datetime64[ns]")
    score_mask = (times >= start) & (times < end + np.timedelta64(1, "D"))
    if not np.any(score_mask):
        raise ValueError("Forecast cache contains no initializations in the scored period.")
    scored = prediction.isel(init_time=np.flatnonzero(score_mask))
    expected_times = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]").astype("datetime64[ns]")
    if not np.array_equal(scored.init_time.values.astype("datetime64[ns]"), expected_times):
        raise ValueError("Scored forecasts must cover every requested daily initialization, with no gaps.")

    warmup_source = prediction
    if args.warmup_cache:
        warmup_source, warmup_leads = forecast_field(args.warmup_cache, variables)
        require_same_coordinates(prediction, warmup_source)
        if not np.array_equal(leads, warmup_leads):
            raise ValueError("Warmup and scored forecasts must have identical lead times.")
    warmup_times = warmup_source.init_time.values.astype("datetime64[ns]")
    keep = np.flatnonzero(warmup_times < start)
    retained = memory_size + int(np.ceil((leads.max() + args.verification_delay_hours) / 24)) + 4
    warmup = warmup_source.isel(init_time=keep[-retained:])
    if not warmup.sizes["init_time"]:
        print("No warmup forecasts were found; the run will include the cold-start period.", file=sys.stderr)

    # Validate every required verification timestamp before starting a long replay.
    all_inits = np.concatenate((warmup.init_time.values, scored.init_time.values)).astype("datetime64[ns]")
    all_valid = all_inits[:, None] + leads.astype("timedelta64[h]")[None, :]
    missing = np.setdiff1d(np.unique(all_valid), truth.time.values.astype("datetime64[ns]"))
    if missing.size:
        raise ValueError(f"Truth is missing {missing.size} required dates, starting with {missing[0]}.")
    metrics = PooledMetrics(len(leads), len(variables), prediction.lat.values)
    replay_rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for phase, source in (("warmup", warmup), ("test", scored)):
            for index, current in enumerate(source.init_time.values.astype("datetime64[ns]")):
                raw = np.asarray(source.isel(init_time=index).values, dtype=np.float32)
                valid = current + leads.astype("timedelta64[h]")
                target = np.asarray(truth.sel(time=valid).values, dtype=np.float32)
                if not np.isfinite(raw).all() or not np.isfinite(target).all():
                    raise ValueError(f"Non-finite forecast or target at {current}.")
                if phase == "test":
                    corrected, gates, counts = calibrator.calibrate(current, raw)
                    if args.require_warmup and index == 0 and any(count < memory_size for count in counts):
                        raise ValueError("Warmup does not provide a full eligible memory for every lead.")
                    reference = np.asarray(climatology.sel(dayofyear=calendar_day_numbers(valid)).values, dtype=np.float32)
                    corrected = corrected.detach().cpu().numpy()
                    metrics.update(raw, corrected, target, reference)
                    replay_rows.append({
                        "init_time": np.datetime_as_string(current, unit="D"),
                        "minimum_memory_pairs": min(counts),
                        "maximum_memory_pairs": max(counts),
                        "mean_gate": float(np.asarray(gates).mean()),
                    })
                calibrator.update(current, raw, target)

    scores = metrics.scores()
    macro = macro_scores(scores, sigma)
    result = {
        "model": args.model, "variables": variables, "lead_time_hours": leads.tolist(),
        "start": str(start), "end": str(end), "n_initializations": metrics.count,
        "initialization_sha256": hashlib.sha256(expected_times.tobytes()).hexdigest(),
        "verification_delay_hours": args.verification_delay_hours,
        "admission_protocol": "archived_target_date" if args.verification_delay_hours == 0 else "target_date_plus_explicit_delay",
        "memory_size": memory_size, "parameters": parameters,
        "parameters_sha256": file_digest(args.params), "statistics_sha256": file_digest(args.statistics),
        "climatology_metadata": climatology_info, "source_kinds": source_kinds,
        "warmup_initializations": warmup.sizes["init_time"],
        "grid": [prediction.sizes["lat"], prediction.sizes["lon"]],
        "device": args.device, "elapsed_seconds": time.perf_counter() - started,
        "metrics": {method: {key: array.tolist() for key, array in values.items()} for method, values in scores.items()},
        "macro": macro,
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "per_variable.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "lead_time_hours", "variable", "raw_rmse", "spherettc_rmse", "raw_acc", "spherettc_acc"])
        writer.writeheader()
        for lead_index, lead in enumerate(leads):
            for variable_index, variable in enumerate(variables):
                writer.writerow({
                    "model": args.model, "lead_time_hours": int(lead), "variable": variable,
                    **{f"{method}_{metric}": float(scores[method][metric][lead_index, variable_index]) for method in scores for metric in ("rmse", "acc")},
                })
    with (output / "replay_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0]))
        writer.writeheader()
        writer.writerows(replay_rows)
    compact = {f"{method}_{key}": array for method, values in metrics.sums.items() for key, array in values.items()}
    np.savez_compressed(output / "sufficient_statistics.npz", n_initializations=metrics.count, **compact)
    raw, corrected = macro["raw"], macro["spherettc"]
    nrmse_gain, acc_gain = relative_improvements(raw["overall_nrmse"], corrected["overall_nrmse"], raw["overall_acc"], corrected["overall_acc"])
    print(json.dumps({"output": str(output), "n_initializations": metrics.count, "nrmse_improvement_percent": nrmse_gain, "acc_improvement_percent": acc_gain}, indent=2))


if __name__ == "__main__":
    main()
