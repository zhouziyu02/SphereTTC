#!/usr/bin/env python
"""Search existing SphereTTC parameters on 2017 and freeze a family/lead schedule."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def find_repository() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts/run_ttc.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the migrated SOON repository root")


SOURCE = find_repository()
RUN_TTC = SOURCE / "scripts/run_ttc.py"
CONFIG = SOURCE / "configs/unified_11model_ttc.yaml"
CACHE = (
    SOURCE
    / "artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_2017_121x240.zarr"
)
TRUTH = SOURCE / "data/S2S"
CLIMATOLOGY = (
    SOURCE
    / "artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz"
)
DESIGN = ROOT / "configs/search_space.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidates() -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "memory_size": 64,
        "lmax": 40,
        "mmax": 40,
        "half_life": 32.0,
        "ridge": 0.05,
        "strength": 0.65,
        "min_memory": 24,
        "holdout_fraction": 0.25,
        "min_holdout": 8,
        "confidence_z": 0.5,
        "robust_quantile_scale": 3.0,
        "max_scale_delta": 0.5,
        "taper_power": 1.0,
    }
    proposed: list[tuple[str, dict[str, Any]]] = []
    for lmax in (16, 24, 32, 40, 48, 64):
        for strength in (0.35, 0.5, 0.65, 0.8, 1.0):
            proposed.append(
                (f"l{lmax}_s{str(strength).replace('.', '')}", {"lmax": lmax, "mmax": lmax, "strength": strength})
            )
    for memory, half_life in ((32, 16.0), (96, 48.0), (128, 64.0), (192, 96.0)):
        proposed.append((f"memory{memory}_half{int(half_life)}", {"memory_size": memory, "half_life": half_life}))
    for confidence in (0.0, 0.25, 0.75, 1.0):
        proposed.append((f"confidence{str(confidence).replace('.', '')}", {"confidence_z": confidence}))
    for taper in (0.25, 0.5, 2.0):
        proposed.append((f"taper{str(taper).replace('.', '')}", {"taper_power": taper}))
    for mmax in (16, 24, 32):
        proposed.append((f"l40_m{mmax}", {"mmax": mmax}))
    for ridge in (0.01, 0.2, 1.0):
        proposed.append((f"ridge{str(ridge).replace('.', '')}", {"ridge": ridge}))
    for delta in (0.1, 0.25, 1.0):
        proposed.append((f"delta{str(delta).replace('.', '')}", {"max_scale_delta": delta}))
    for fraction, minimum in ((0.15, 6), (0.33, 12), (0.5, 16)):
        proposed.append((f"holdout{str(fraction).replace('.', '')}", {"holdout_fraction": fraction, "min_holdout": minimum}))
    for minimum in (16, 32, 48):
        proposed.append((f"minmemory{minimum}", {"min_memory": minimum}))

    unique: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for candidate_id, overrides in proposed:
        params = base | overrides
        canonical = json.dumps(params, sort_keys=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique[candidate_id] = {"id": candidate_id, **params}
    return list(unique.values())


def write_params(items: list[dict[str, Any]]) -> None:
    output = ROOT / "configs/generated"
    output.mkdir(parents=True, exist_ok=True)
    for item in items:
        payload = {"best_params": {key: value for key, value in item.items() if key != "id"}}
        (output / f"{item['id']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    (ROOT / "configs/candidate_manifest.json").write_text(
        json.dumps({"count": len(items), "candidates": items}, indent=2) + "\n"
    )


def base_command(metrics: Path, profile: Path) -> list[str]:
    return [
        sys.executable,
        str(RUN_TTC),
        "--config",
        str(CONFIG),
        "--model",
        "graphcast",
        "--cache",
        str(CACHE),
        "--truth-root",
        str(TRUTH),
        "--truth-workers",
        "4",
        "--acc-climatology",
        str(CLIMATOLOGY),
        "--no-cache-output",
        "--metrics-output",
        str(metrics),
        "--profile-output",
        str(profile),
    ]


def execute(task: str, command: list[str], gpu: int) -> None:
    log_path = ROOT / "logs" / f"{task}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w") as log:
        log.write(f"started_at={utc_now()}\nphysical_gpu={gpu}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=SOURCE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        log.write(f"finished_at={utc_now()}\nreturncode={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"{task} failed; see {log_path}")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def variable_family(variable: str) -> str:
    if variable == "mslp":
        return "mslp"
    if variable == "t2m" or (variable.startswith("t") and variable[1:].isdigit()):
        return "temperature"
    if variable == "u10" or (variable.startswith("u") and variable[1:].isdigit()):
        return "zonal_wind"
    if variable == "v10" or (variable.startswith("v") and variable[1:].isdigit()):
        return "meridional_wind"
    if variable.startswith("z"):
        return "geopotential"
    if variable.startswith("q"):
        return "humidity"
    raise ValueError(variable)


def aggregate(data: dict[str, np.ndarray], start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    subset = slice(start, stop)
    rmse = np.sqrt(data["mse"][subset].mean(axis=0))
    numerator = data["acc_numerator"][subset].sum(axis=0)
    prediction = data["acc_prediction_energy"][subset].sum(axis=0)
    target = data["acc_target_energy"][subset].sum(axis=0)
    acc = numerator / np.sqrt(np.maximum(prediction * target, 1e-24))
    return rmse, acc


def window_cell_metrics(
    raw: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rmse, raw_acc = aggregate(raw, start, stop)
    ttc_rmse, ttc_acc = aggregate(calibrated, start, stop)
    return 1.0 - ttc_rmse / raw_rmse, ttc_acc - raw_acc


def combine_schedule(
    raw: dict[str, np.ndarray],
    metrics: dict[str, dict[str, np.ndarray]],
    schedule: dict[str, dict[str, str]],
) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in raw.items()}
    variables = [str(value) for value in raw["variable"]]
    for lead_index, lead in enumerate(raw["lead_time"]):
        lead_schedule = schedule[str(int(lead))]
        for variable_index, variable in enumerate(variables):
            candidate_id = lead_schedule[variable_family(variable)]
            if candidate_id == "noop":
                continue
            source = metrics[candidate_id]
            for name in (
                "mse",
                "mae",
                "bias",
                "calibration_gate",
                "acc_numerator",
                "acc_prediction_energy",
                "acc_target_energy",
            ):
                result[name][:, lead_index, variable_index] = source[name][
                    :, lead_index, variable_index
                ]
    return result


def summarize(
    raw: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    start: int,
    stop: int,
) -> dict[str, Any]:
    gain, acc_delta = window_cell_metrics(raw, calibrated, start, stop)
    return {
        "initializations": stop - start,
        "mean_relative_rmse_gain": float(gain.mean()),
        "median_relative_rmse_gain": float(np.median(gain)),
        "minimum_relative_rmse_gain": float(gain.min()),
        "negative_rmse_cells": int((gain < 0).sum()),
        "negative_rmse_cell_fraction": float((gain < 0).mean()),
        "mean_acc_delta": float(acc_delta.mean()),
        "minimum_acc_delta": float(acc_delta.min()),
        "negative_acc_cells": int((acc_delta < 0).sum()),
    }


def select_schedule(items: list[dict[str, Any]]) -> dict[str, Any]:
    design = json.loads(DESIGN.read_text())
    search_start, search_stop = design["search_initialization_slice"]
    holdout_start, holdout_stop = design["internal_holdout_slice"]
    guards = design["search_eligibility"]
    raw = load(ROOT / "metrics/validation/raw.npz")
    metric_data = {
        item["id"]: load(ROOT / f"metrics/validation/{item['id']}.npz")
        for item in items
    }
    variables = [str(value) for value in raw["variable"]]
    families = sorted({variable_family(variable) for variable in variables})
    family_indices = {
        family: np.asarray(
            [index for index, variable in enumerate(variables) if variable_family(variable) == family]
        )
        for family in families
    }
    cell_metrics = {
        candidate_id: window_cell_metrics(raw, data, search_start, search_stop)
        for candidate_id, data in metric_data.items()
    }
    schedule: dict[str, dict[str, str]] = {}
    selected_rows: list[dict[str, Any]] = []
    for lead_index, lead in enumerate(raw["lead_time"]):
        lead_key = str(int(lead))
        schedule[lead_key] = {}
        for family in families:
            indices = family_indices[family]
            best: tuple[float, float, str, dict[str, Any]] | None = None
            for item in items:
                gain, acc_delta = cell_metrics[item["id"]]
                current_gain = gain[lead_index, indices]
                current_acc = acc_delta[lead_index, indices]
                row = {
                    "lead_time_hours": int(lead),
                    "family": family,
                    "candidate_id": item["id"],
                    "mean_relative_rmse_gain": float(current_gain.mean()),
                    "negative_rmse_cell_fraction": float((current_gain < 0).mean()),
                    "mean_acc_delta": float(current_acc.mean()),
                }
                eligible = bool(
                    row["mean_relative_rmse_gain"] > guards["minimum_mean_relative_rmse_gain"]
                    and row["negative_rmse_cell_fraction"] <= guards["maximum_negative_rmse_cell_fraction"]
                    and row["mean_acc_delta"] >= guards["minimum_mean_acc_delta"]
                )
                if not eligible:
                    continue
                key = (row["mean_relative_rmse_gain"], row["mean_acc_delta"], item["id"], row)
                if best is None or key[:2] > best[:2]:
                    best = key
            if best is None:
                candidate_id = "noop"
                chosen = {
                    "lead_time_hours": int(lead),
                    "family": family,
                    "candidate_id": "noop",
                    "mean_relative_rmse_gain": 0.0,
                    "negative_rmse_cell_fraction": 0.0,
                    "mean_acc_delta": 0.0,
                }
            else:
                candidate_id = best[2]
                chosen = best[3]
            schedule[lead_key][family] = candidate_id
            selected_rows.append(chosen)

    combined = combine_schedule(raw, metric_data, schedule)
    internal = summarize(raw, combined, holdout_start, holdout_stop)
    full_development = summarize(raw, combined, search_start, holdout_stop)
    selected_ids = sorted(
        {candidate_id for mapping in schedule.values() for candidate_id in mapping.values() if candidate_id != "noop"}
    )
    result = {
        "status": "FROZEN_BEFORE_2020",
        "created_at": utc_now(),
        "prospective_2020_metrics_touched": False,
        "search_slice": [search_start, search_stop],
        "internal_holdout_slice": [holdout_start, holdout_stop],
        "schedule": schedule,
        "selected_candidate_ids": selected_ids,
        "selected_rows": selected_rows,
        "internal_holdout_summary": internal,
        "full_development_summary": full_development,
    }
    (ROOT / "FROZEN_SCHEDULE_2017.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "selected_schedule.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    print(
        "SCHEDULE_FROZEN",
        f"unique_candidates={len(selected_ids)}",
        f"holdout_rmse_gain={100 * internal['mean_relative_rmse_gain']:.4f}%",
        f"holdout_acc_delta={internal['mean_acc_delta']:+.8f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    items = candidates()
    write_params(items)
    print(f"CANDIDATE_COUNT {len(items)}", flush=True)
    if not args.score_only:
        execute(
            "validation_raw",
            base_command(ROOT / "metrics/validation/raw.npz", ROOT / "profiles/validation_raw.json")
            + ["--method", "raw"],
            args.gpu,
        )
        for index, item in enumerate(items, start=1):
            task = f"validation_{item['id']}"
            execute(
                task,
                base_command(
                    ROOT / f"metrics/validation/{item['id']}.npz",
                    ROOT / f"profiles/{task}.json",
                )
                + [
                    "--method",
                    "sphere_ttc",
                    "--params-json",
                    str(ROOT / f"configs/generated/{item['id']}.json"),
                ],
                args.gpu,
            )
            print(f"VALIDATION_CANDIDATE_COMPLETE {index}/{len(items)} {item['id']}", flush=True)
    select_schedule(items)


if __name__ == "__main__":
    main()
