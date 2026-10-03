#!/usr/bin/env python
"""Run an isolated parameter-only GraphCast + SphereTTC GPU experiment."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CACHE_ROOT = (
    SOURCE
    / "artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast"
)
TRUTH_ROOT = SOURCE / "data/S2S"
CLIMATOLOGY = (
    SOURCE
    / "artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz"
)
CANDIDATE_FILE = ROOT / "configs/candidates.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_design() -> dict[str, Any]:
    return json.loads(CANDIDATE_FILE.read_text())


def candidate_params(design: dict[str, Any]) -> list[dict[str, Any]]:
    fixed = design["fixed"]
    return [fixed | candidate for candidate in design["candidates"]]


def gpu_inventory(label: str) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
        "--format=csv,noheader",
    ]
    text = subprocess.check_output(command, text=True)
    (ROOT / f"gpu_{label}.txt").write_text(text)


def write_generated_params(candidates: list[dict[str, Any]]) -> None:
    output = ROOT / "configs/generated"
    output.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        payload = {"best_params": {key: value for key, value in candidate.items() if key != "id"}}
        (output / f"{candidate['id']}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def base_command(cache: Path, metrics: Path, profile: Path) -> list[str]:
    return [
        sys.executable,
        str(RUN_TTC),
        "--config",
        str(CONFIG),
        "--model",
        "graphcast",
        "--cache",
        str(cache),
        "--truth-root",
        str(TRUTH_ROOT),
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


def execute(task_id: str, gpu: int, command: list[str]) -> dict[str, Any]:
    log_path = ROOT / "logs" / f"{task_id}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = utc_now()
    with log_path.open("w") as log:
        log.write(f"started_at={started}\n")
        log.write(f"physical_gpu={gpu}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=SOURCE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"finished_at={utc_now()}\n")
        log.write(f"returncode={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"{task_id} failed; see {log_path}")
    return {"task": task_id, "gpu": gpu, "log": str(log_path)}


def load_metric(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def concatenate(paths: list[Path]) -> dict[str, np.ndarray]:
    items = [load_metric(path) for path in paths]
    for key in ("lead_time", "variable"):
        if not all(np.array_equal(items[0][key], item[key]) for item in items[1:]):
            raise RuntimeError(f"metric coordinate mismatch: {key}")
    result = {"lead_time": items[0]["lead_time"], "variable": items[0]["variable"]}
    for key in (
        "init_time",
        "mse",
        "mae",
        "bias",
        "eligible_memory_count",
        "calibration_gate",
        "acc_numerator",
        "acc_prediction_energy",
        "acc_target_energy",
    ):
        result[key] = np.concatenate([item[key] for item in items], axis=0)
    return result


def paired_summary(
    raw: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for key in ("init_time", "lead_time", "variable"):
        if not np.array_equal(raw[key], calibrated[key]):
            raise RuntimeError(f"paired metric mismatch: {key}")
    keep = slice(start, None)
    raw_rmse = np.sqrt(raw["mse"][keep].mean(axis=0))
    calibrated_rmse = np.sqrt(calibrated["mse"][keep].mean(axis=0))
    gain = 1.0 - calibrated_rmse / raw_rmse

    def acc(data: dict[str, np.ndarray]) -> np.ndarray:
        numerator = data["acc_numerator"][keep].sum(axis=0)
        energy = data["acc_prediction_energy"][keep].sum(axis=0)
        target = data["acc_target_energy"][keep].sum(axis=0)
        return numerator / np.sqrt(np.maximum(energy * target, 1e-24))

    raw_acc = acc(raw)
    calibrated_acc = acc(calibrated)
    acc_delta = calibrated_acc - raw_acc
    details: list[dict[str, Any]] = []
    for lead_index, lead in enumerate(raw["lead_time"]):
        for variable_index, variable in enumerate(raw["variable"]):
            details.append(
                {
                    "lead_time_hours": int(lead),
                    "variable": str(variable),
                    "raw_rmse": float(raw_rmse[lead_index, variable_index]),
                    "sphere_ttc_rmse": float(calibrated_rmse[lead_index, variable_index]),
                    "relative_rmse_gain": float(gain[lead_index, variable_index]),
                    "raw_acc": float(raw_acc[lead_index, variable_index]),
                    "sphere_ttc_acc": float(calibrated_acc[lead_index, variable_index]),
                    "acc_delta": float(acc_delta[lead_index, variable_index]),
                }
            )
    by_lead = {}
    for lead_index, lead in enumerate(raw["lead_time"]):
        current_gain = gain[lead_index]
        current_acc = acc_delta[lead_index]
        by_lead[str(int(lead))] = {
            "mean_relative_rmse_gain": float(current_gain.mean()),
            "negative_rmse_cells": int((current_gain < 0).sum()),
            "mean_acc_delta": float(current_acc.mean()),
            "negative_acc_cells": int((current_acc < 0).sum()),
        }
    summary = {
        "scored_initializations": int(raw["init_time"][keep].size),
        "cell_count": int(gain.size),
        "mean_relative_rmse_gain": float(gain.mean()),
        "median_relative_rmse_gain": float(np.median(gain)),
        "minimum_relative_rmse_gain": float(gain.min()),
        "negative_rmse_cells": int((gain < 0).sum()),
        "negative_rmse_cell_fraction": float((gain < 0).mean()),
        "mean_acc_delta": float(acc_delta.mean()),
        "minimum_acc_delta": float(acc_delta.min()),
        "negative_acc_cells": int((acc_delta < 0).sum()),
        "mean_calibration_gate": float(calibrated["calibration_gate"][keep].mean()),
        "by_lead": by_lead,
    }
    return summary, details


def run_validation() -> dict[str, Any]:
    design = load_design()
    candidates = candidate_params(design)
    write_generated_params(candidates)
    raw_metrics = ROOT / "metrics/validation/raw.npz"
    raw_profile = ROOT / "profiles/validation_raw.json"
    execute(
        "validation_raw",
        0,
        base_command(CACHE_ROOT / "forecast_cache_2017_121x240.zarr", raw_metrics, raw_profile)
        + ["--method", "raw"],
    )
    tasks = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, candidate in enumerate(candidates):
            candidate_id = candidate["id"]
            metrics = ROOT / f"metrics/validation/{candidate_id}.npz"
            profile = ROOT / f"profiles/validation_{candidate_id}.json"
            command = base_command(
                CACHE_ROOT / "forecast_cache_2017_121x240.zarr", metrics, profile
            ) + [
                "--method",
                "sphere_ttc",
                "--params-json",
                str(ROOT / f"configs/generated/{candidate_id}.json"),
            ]
            tasks.append(executor.submit(execute, f"validation_{candidate_id}", index % 2, command))
        for future in as_completed(tasks):
            future.result()

    raw = load_metric(raw_metrics)
    scored = []
    for candidate in candidates:
        candidate_id = candidate["id"]
        calibrated = load_metric(ROOT / f"metrics/validation/{candidate_id}.npz")
        summary, _ = paired_summary(
            raw,
            calibrated,
            int(design["score_after_initialization"]),
        )
        guards = design["validation_guards"]
        eligible = bool(
            summary["negative_rmse_cell_fraction"]
            <= guards["maximum_negative_rmse_cell_fraction"]
            and summary["mean_acc_delta"] >= guards["minimum_macro_acc_delta"]
        )
        scored.append({"id": candidate_id, "params": {k: v for k, v in candidate.items() if k != "id"}, "eligible": eligible, **summary})
    eligible = [item for item in scored if item["eligible"]]
    fallback_used = not eligible
    pool = eligible if eligible else scored
    best = max(
        pool,
        key=lambda item: (
            item["mean_relative_rmse_gain"],
            item["mean_acc_delta"],
            -item["negative_rmse_cell_fraction"],
        ),
    )
    selection = {
        "created_at": utc_now(),
        "test_data_touched_by_selection": False,
        "fallback_used": fallback_used,
        "best_id": best["id"],
        "best_params": best["params"],
        "validation_guards": design["validation_guards"],
        "candidates": scored,
    }
    (ROOT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def test_commands(selected_id: str) -> list[tuple[str, int, list[str]]]:
    commands: list[tuple[str, int, list[str]]] = []
    for gpu, year in enumerate((2018, 2019)):
        cache = CACHE_ROOT / f"forecast_cache_{year}_121x240.zarr"
        raw_metrics = ROOT / f"metrics/test/raw_{year}.npz"
        raw_profile = ROOT / f"profiles/test_raw_{year}.json"
        commands.append(
            (
                f"test_raw_{year}",
                gpu,
                base_command(cache, raw_metrics, raw_profile) + ["--method", "raw"],
            )
        )
    for gpu, year in enumerate((2018, 2019)):
        cache = CACHE_ROOT / f"forecast_cache_{year}_121x240.zarr"
        warmup_year = year - 1
        metrics = ROOT / f"metrics/test/sphere_ttc_{year}.npz"
        profile = ROOT / f"profiles/test_sphere_ttc_{year}.json"
        command = base_command(cache, metrics, profile) + [
            "--method",
            "sphere_ttc",
            "--params-json",
            str(ROOT / f"configs/generated/{selected_id}.json"),
            "--warmup-cache",
            str(CACHE_ROOT / f"forecast_cache_{warmup_year}_121x240.zarr"),
            "--trim-warmup-cache",
            "--warmup-truth-root",
            str(TRUTH_ROOT),
        ]
        commands.append((f"test_sphere_ttc_{year}", gpu, command))
    return commands


def render_report(selection: dict[str, Any], summary: dict[str, Any], expectation: dict[str, Any]) -> None:
    primary = bool(
        summary["mean_relative_rmse_gain"] >= expectation["minimum_mean_relative_rmse_gain"]
        and summary["negative_rmse_cell_fraction"]
        <= expectation["maximum_negative_rmse_cell_fraction"]
    )
    strict_acc = bool(summary["mean_acc_delta"] >= expectation["strict_minimum_macro_acc_delta"])
    lines = [
        "# GraphCast + SphereTTC parameter-only GPU result",
        "",
        f"- Selected on 2017 only: `{selection['best_id']}`",
        f"- Frozen test: 2018-2019 ({summary['scored_initializations']} daily initializations)",
        f"- Mean cell-relative RMSE gain: {100.0 * summary['mean_relative_rmse_gain']:.4f}%",
        f"- RMSE-regressed cells: {summary['negative_rmse_cells']}/{summary['cell_count']} ({100.0 * summary['negative_rmse_cell_fraction']:.4f}%)",
        f"- Macro ACC change: {summary['mean_acc_delta']:+.8f}",
        f"- ACC-regressed cells: {summary['negative_acc_cells']}/{summary['cell_count']}",
        f"- Primary expected effect: {'PASS' if primary else 'FAIL'}",
        f"- Strict non-decreasing macro ACC: {'PASS' if strict_acc else 'FAIL'}",
        "",
        "## By lead",
        "",
        "| Lead | Mean RMSE gain | RMSE regressions | Mean ACC change | ACC regressions |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lead, row in summary["by_lead"].items():
        lines.append(
            f"| {lead}h | {100.0 * row['mean_relative_rmse_gain']:.4f}% | "
            f"{row['negative_rmse_cells']}/49 | {row['mean_acc_delta']:+.8f} | "
            f"{row['negative_acc_cells']}/49 |"
        )
    lines.extend(
        [
            "",
            "The test result is paired against raw GraphCast metrics recomputed in this",
            "archived experiment directory. Algorithm source and frozen caches are",
            "resolved from the enclosing portable SOON repository.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_test() -> dict[str, Any]:
    design = load_design()
    selection = json.loads((ROOT / "selection.json").read_text())
    if selection["fallback_used"]:
        print("WARNING: no validation candidate met every guard; running diagnostic fallback")
    commands = test_commands(selection["best_id"])
    for offset in (0, 2):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute, *task) for task in commands[offset : offset + 2]]
            for future in as_completed(futures):
                future.result()
    raw = concatenate([ROOT / "metrics/test/raw_2018.npz", ROOT / "metrics/test/raw_2019.npz"])
    calibrated = concatenate(
        [ROOT / "metrics/test/sphere_ttc_2018.npz", ROOT / "metrics/test/sphere_ttc_2019.npz"]
    )
    summary, details = paired_summary(raw, calibrated, 0)
    expectation = design["test_expectation"]
    summary["selected_id"] = selection["best_id"]
    summary["selected_params"] = selection["best_params"]
    summary["primary_expected_effect_pass"] = bool(
        summary["mean_relative_rmse_gain"] >= expectation["minimum_mean_relative_rmse_gain"]
        and summary["negative_rmse_cell_fraction"]
        <= expectation["maximum_negative_rmse_cell_fraction"]
    )
    summary["strict_acc_pass"] = bool(
        summary["mean_acc_delta"] >= expectation["strict_minimum_macro_acc_delta"]
    )
    summary["test_expectation"] = expectation
    (ROOT / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "test_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    render_report(selection, summary, expectation)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("validation", "test", "all"))
    args = parser.parse_args()
    gpu_inventory("before")
    if args.phase in ("validation", "all"):
        selection = run_validation()
        print(
            "SELECTED",
            selection["best_id"],
            "fallback_used=",
            selection["fallback_used"],
            flush=True,
        )
    if args.phase in ("test", "all"):
        result = run_test()
        print(
            "TEST",
            "primary_pass=",
            result["primary_expected_effect_pass"],
            "strict_acc_pass=",
            result["strict_acc_pass"],
            flush=True,
        )
    gpu_inventory("after")


if __name__ == "__main__":
    main()
