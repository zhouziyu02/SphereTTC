#!/usr/bin/env python
"""Fit and apply the frozen single-backbone SphereTTC-v22 main prediction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
import xarray as xr
import zarr

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.spheredyn.evaluate_goal_reference_ensemble import (
    REFERENCES,
    _fit_backbone,
    _open,
    _prediction,
    _target,
    _validate_coordinates,
)
from scripts.spheredyn.infer_spatiotemporal_checkpoint import _create_cache
from src.experiments.protocol import COMMON_EVALUATION_VARIABLES
from src.utils.device import get_device_info, require_cuda


def _year_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected YEAR=PATH")
    year, path = value.split("=", 1)
    return int(year), Path(path)


def _open_many(paths: list[Path], variables: list[str]) -> xr.Dataset:
    datasets = [_open(path, variables) for path in paths]
    combined = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="init_time")
    combined = combined.sortby("init_time")
    init_time = np.asarray(combined.init_time.values)
    if len(np.unique(init_time)) != len(init_time):
        raise RuntimeError("backbone fragments contain duplicate initialization dates")
    return combined


def _empty_statistics(
    days: int,
    leads: int,
    variable_count: int,
    latitude_count: int,
    longitude_count: int,
    reference_count: int,
    sample_latitude: np.ndarray,
    sample_stride: int,
) -> dict[str, np.ndarray]:
    return {
        "days": np.asarray(days),
        "count": np.asarray(
            float(days)
            * float(sample_latitude.sum())
            * float(len(np.arange(0, longitude_count, sample_stride)))
        ),
        "reference_cross": np.zeros(
            (leads, variable_count, reference_count, reference_count)
        ),
        "reference_sum": np.zeros((leads, variable_count, reference_count)),
        "reference_map_sum": np.zeros(
            (
                leads,
                variable_count,
                latitude_count,
                longitude_count,
                reference_count,
            )
        ),
        "raw_reference_cross": np.zeros(
            (1, leads, variable_count, reference_count)
        ),
        "raw_sse": np.zeros((1, leads, variable_count)),
        "raw_sum": np.zeros((1, leads, variable_count)),
        "raw_map_sum": np.zeros(
            (1, leads, variable_count, latitude_count, longitude_count)
        ),
    }


def _fit(
    backbone: xr.Dataset,
    references: dict[str, xr.Dataset],
    sample_stride: int,
    validation_days: int,
    batch_size: int,
    minimum_backbone_weight: float,
) -> dict[str, np.ndarray]:
    for name, dataset in references.items():
        _validate_coordinates(backbone, dataset, f"2017 reference {name}")
    total_days = backbone.sizes["init_time"]
    validation_days = min(validation_days, total_days // 2)
    train_days = total_days - validation_days
    latitude = np.cos(np.deg2rad(np.asarray(backbone.lat.values, dtype=np.float64)))
    sample_latitude = latitude[::sample_stride]
    shape = {
        "leads": backbone.sizes["lead_time"],
        "variable_count": backbone.sizes["variable"],
        "latitude_count": backbone.sizes["lat"],
        "longitude_count": backbone.sizes["lon"],
        "reference_count": len(REFERENCES),
        "sample_latitude": sample_latitude,
        "sample_stride": sample_stride,
    }
    statistics = {
        "train": _empty_statistics(train_days, **shape),
        "validation": _empty_statistics(validation_days, **shape),
    }
    for start in range(0, total_days, batch_size):
        stop = min(start + batch_size, total_days)
        subset = "train" if start < train_days else "validation"
        if start < train_days < stop:
            raise RuntimeError("fit batch crosses the train/validation boundary")
        indexer = slice(start, stop)
        target = _target(backbone, indexer)
        reference_predictions = np.stack(
            [_prediction(references[name], indexer) for name in REFERENCES], axis=0
        )
        reference_errors = reference_predictions - target[None]
        stats = statistics[subset]
        stats["reference_map_sum"] += reference_errors.sum(axis=1).transpose(
            1, 2, 3, 4, 0
        )
        sampled_reference = reference_errors[
            ..., ::sample_stride, ::sample_stride
        ]
        stats["reference_sum"] += np.einsum(
            "rblvxy,x->lvr", sampled_reference, sample_latitude, optimize=True
        )
        weighted_reference = sampled_reference * np.sqrt(sample_latitude)[
            None, None, None, None, :, None
        ]
        reference_matrix = weighted_reference.transpose(2, 3, 1, 4, 5, 0).reshape(
            shape["leads"], shape["variable_count"], -1, shape["reference_count"]
        )
        stats["reference_cross"] += np.einsum(
            "lvsm,lvsn->lvmn", reference_matrix, reference_matrix, optimize=True
        )
        raw_error = _prediction(backbone, indexer) - target
        stats["raw_map_sum"][0] += raw_error.sum(axis=0)
        sampled_raw = raw_error[..., ::sample_stride, ::sample_stride]
        stats["raw_sum"][0] += np.einsum(
            "blvxy,x->lv", sampled_raw, sample_latitude, optimize=True
        )
        weighted_raw = sampled_raw * np.sqrt(sample_latitude)[None, None, None, :, None]
        raw_matrix = weighted_raw.transpose(1, 2, 0, 3, 4).reshape(
            shape["leads"], shape["variable_count"], -1
        )
        stats["raw_sse"][0] += np.einsum(
            "lvs,lvs->lv", raw_matrix, raw_matrix, optimize=True
        )
        stats["raw_reference_cross"][0] += np.einsum(
            "lvs,lvsm->lvm", raw_matrix, reference_matrix, optimize=True
        )
        if stop % 32 == 0 or stop == total_days:
            print(f"fit SphereTTC statistics {stop}/{total_days}", flush=True)
    return _fit_backbone(
        0,
        statistics["train"],
        statistics["validation"],
        latitude,
        minimum_backbone_weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-backbone-fragment", action="append", type=Path, required=True)
    parser.add_argument("--eval-backbone-fragment", action="append", type=_year_path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fit-sample-stride", type=int, default=4)
    parser.add_argument("--validation-days", type=int, default=121)
    parser.add_argument("--minimum-backbone-weight", type=float, default=0.5)
    args = parser.parse_args()

    device = require_cuda("v2 main SphereTTC application")
    variables = list(COMMON_EVALUATION_VARIABLES)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fit_backbone = _open_many(args.fit_backbone_fragment, variables)
    fit_references = {
        name: _open(
            args.reference_root / name / "forecast_cache_2017_121x240.zarr",
            variables,
        )
        for name in REFERENCES
    }
    fitted = _fit(
        fit_backbone,
        fit_references,
        args.fit_sample_stride,
        args.validation_days,
        args.batch_size,
        args.minimum_backbone_weight,
    )
    np.savez_compressed(
        output_root / "FITTED_WEIGHTS_2017.npz",
        **fitted,
    )

    stats = json.loads(args.stats.read_text())
    std_lookup = dict(zip(stats["variables"], stats["std"], strict=True))
    standard_deviation = np.asarray([std_lookup[name] for name in variables])
    latitude = np.cos(np.deg2rad(np.asarray(fit_backbone.lat.values, dtype=np.float64)))
    latitude_tensor = torch.as_tensor(latitude, dtype=torch.float64, device=device)
    weights = torch.as_tensor(fitted["weights"], dtype=torch.float64, device=device)
    bias = torch.as_tensor(fitted["bias"], dtype=torch.float64, device=device)
    spatial_bias = torch.as_tensor(
        fitted["spatial_bias"], dtype=torch.float64, device=device
    )
    eval_paths: dict[int, list[Path]] = {}
    for year, path in args.eval_backbone_fragment:
        eval_paths.setdefault(year, []).append(path)

    combined_raw_mse = []
    combined_final_mse = []
    combined_graphcast_mse = []
    combined_init_time = []
    for year in sorted(eval_paths):
        raw_dataset = _open_many(eval_paths[year], variables)
        references = {
            name: _open(
                args.reference_root / name / f"forecast_cache_{year}_121x240.zarr",
                variables,
            ).sel(init_time=raw_dataset.init_time.values)
            for name in REFERENCES
        }
        for name, dataset in references.items():
            _validate_coordinates(raw_dataset, dataset, f"{year} reference {name}")
        final_path = output_root / f"final_spheredyn_v9_spherettc_v22_{year}.zarr"
        group = _create_cache(
            final_path,
            np.asarray(raw_dataset.init_time.values),
            np.asarray(raw_dataset.lead_time.values),
            variables,
            np.asarray(raw_dataset.lat.values),
            np.asarray(raw_dataset.lon.values),
            {
                "model_name": "spheredyn_v9_seed44_plus_spherettc_v22",
                "source_type": "frozen_main_prediction_v2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "minimum_backbone_weight": str(args.minimum_backbone_weight),
                "device_info": str(get_device_info()),
            },
            include_target=True,
        )
        days = raw_dataset.sizes["init_time"]
        raw_year_mse = np.zeros((days, len(raw_dataset.lead_time), len(variables)))
        final_year_mse = np.zeros_like(raw_year_mse)
        graphcast_year_mse = np.zeros_like(raw_year_mse)
        denominator = float(latitude.sum()) * float(raw_dataset.sizes["lon"])
        for start in range(0, days, args.batch_size):
            stop = min(start + args.batch_size, days)
            indexer = slice(start, stop)
            target_cpu = _target(raw_dataset, indexer)
            raw_cpu = _prediction(raw_dataset, indexer)
            reference_cpu = [
                _prediction(references[name], indexer) for name in REFERENCES
            ]
            target = torch.as_tensor(target_cpu, dtype=torch.float64, device=device)
            raw = torch.as_tensor(raw_cpu, dtype=torch.float64, device=device)
            corrected = (
                bias[None, :, :, None, None]
                - spatial_bias[None]
                + raw * weights[None, :, :, -1, None, None]
            )
            for reference_index, prediction in enumerate(reference_cpu):
                corrected = corrected + torch.as_tensor(
                    prediction, dtype=torch.float64, device=device
                ) * weights[None, :, :, reference_index, None, None]
            raw_error = raw - target
            final_error = corrected - target
            graphcast_error = torch.as_tensor(
                reference_cpu[REFERENCES.index("graphcast")],
                dtype=torch.float64,
                device=device,
            ) - target
            raw_year_mse[start:stop] = (
                torch.einsum("blvxy,x->blv", raw_error.square(), latitude_tensor)
                / denominator
            ).cpu().numpy()
            final_year_mse[start:stop] = (
                torch.einsum("blvxy,x->blv", final_error.square(), latitude_tensor)
                / denominator
            ).cpu().numpy()
            graphcast_year_mse[start:stop] = (
                torch.einsum("blvxy,x->blv", graphcast_error.square(), latitude_tensor)
                / denominator
            ).cpu().numpy()
            group["pred"][start:stop] = corrected.float().cpu().numpy()
            group["target"][start:stop] = target_cpu
            if stop % 32 == 0 or stop == days:
                print(f"apply main SphereTTC {year} {stop}/{days}", flush=True)
        zarr.consolidate_metadata(str(final_path))
        combined_raw_mse.append(raw_year_mse)
        combined_final_mse.append(final_year_mse)
        combined_graphcast_mse.append(graphcast_year_mse)
        combined_init_time.append(np.asarray(raw_dataset.init_time.values))

    coordinates = {
        "init_time": np.concatenate(combined_init_time),
        "lead_time": np.asarray(fit_backbone.lead_time.values),
        "variable": np.asarray(variables),
    }
    for name, values in (
        ("RAW_METRICS.npz", combined_raw_mse),
        ("FINAL_METRICS.npz", combined_final_mse),
        ("GRAPHCAST_METRICS.npz", combined_graphcast_mse),
    ):
        np.savez_compressed(output_root / name, mse=np.concatenate(values), **coordinates)
    summary = {
        "status": "MAIN_PREDICTION_CACHES_AND_METRICS_COMPLETE",
        "seed": 44,
        "evaluation_years": sorted(eval_paths),
        "evaluation_days": int(sum(len(values) for values in combined_init_time)),
        "variables": len(variables),
        "minimum_backbone_weight": args.minimum_backbone_weight,
        "device_info": get_device_info(),
        "raw_macro_nrmse": float(
            (np.sqrt(np.concatenate(combined_raw_mse).mean(axis=0)) / standard_deviation[None]).mean()
        ),
        "final_macro_nrmse": float(
            (np.sqrt(np.concatenate(combined_final_mse).mean(axis=0)) / standard_deviation[None]).mean()
        ),
        "prospective_2023_touched": False,
    }
    (output_root / "MAIN_RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
