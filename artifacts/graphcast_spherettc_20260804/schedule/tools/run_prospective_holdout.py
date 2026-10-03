#!/usr/bin/env python
"""Apply the frozen 2017 parameter schedule to the new 2020 GraphCast holdout."""

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
sys.path.insert(0, str(ROOT / "tools"))
import search_schedule as search

RUN_TTC = SOURCE / "scripts/run_ttc.py"
CONFIG = SOURCE / "configs/unified_11model_ttc.yaml"
FORECAST = ROOT / "cache/graphcast_2020_current_a100.zarr"
WARMUP = ROOT / "cache/graphcast_2019_current_a100.zarr"
TRUTH = ROOT / "data/S2S"
WARMUP_TRUTH = SOURCE / "data/S2S"
CLIMATOLOGY = (
    SOURCE
    / "artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_command(metrics: Path, profile: Path) -> list[str]:
    return [
        sys.executable,
        str(RUN_TTC),
        "--config",
        str(CONFIG),
        "--model",
        "graphcast",
        "--cache",
        str(FORECAST),
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


def validate_inputs(schedule: dict[str, Any], primary: dict[str, Any]) -> None:
    if schedule.get("status") != "FROZEN_BEFORE_2020":
        raise RuntimeError("schedule is not frozen")
    if schedule.get("prospective_2020_metrics_touched") is not False:
        raise RuntimeError("schedule lacks prospective isolation declaration")
    if primary.get("status") != "FROZEN_BEFORE_2020_METRICS":
        raise RuntimeError("primary global setting is not frozen")
    if primary.get("prospective_2020_metrics_touched") is not False:
        raise RuntimeError("primary global setting lacks prospective isolation declaration")
    for year in (2019, 2020):
        status = json.loads((ROOT / f"profiles/graphcast_{year}_current_a100.json").read_text())
        expected = 206 if year == 2019 else 366
        if not status.get("complete") or status.get("shape", [0])[0] != expected:
            raise RuntimeError(f"GraphCast {year} cache is incomplete")
        if year == 2019 and not str(status.get("init_start", "")).startswith("2019-06-09"):
            raise RuntimeError("GraphCast 2019 warmup does not contain the required tail")
    truth = json.loads((TRUTH / "HOLDOUT_MANIFEST.json").read_text())
    if truth.get("start") != "2020-01-01" or truth.get("end_inclusive") != "2021-01-10":
        raise RuntimeError("prospective truth manifest has wrong coverage")


def save_composite(
    data: dict[str, np.ndarray], schedule: dict[str, Any], filename: str
) -> None:
    payload = {
        key: value
        for key, value in data.items()
        if key
        in {
            "init_time",
            "lead_time",
            "variable",
            "mse",
            "mae",
            "bias",
            "calibration_gate",
            "acc_numerator",
            "acc_prediction_energy",
            "acc_target_energy",
        }
    }
    payload["schedule_json"] = np.asarray(json.dumps(schedule["schedule"], sort_keys=True))
    np.savez_compressed(ROOT / "metrics/holdout_2020" / filename, **payload)


def cell_details(
    raw: dict[str, np.ndarray], calibrated: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_rmse, raw_acc = search.aggregate(raw, 0, len(raw["init_time"]))
    ttc_rmse, ttc_acc = search.aggregate(calibrated, 0, len(calibrated["init_time"]))
    gain = 1.0 - ttc_rmse / raw_rmse
    delta = ttc_acc - raw_acc
    rows = []
    for lead_index, lead in enumerate(raw["lead_time"]):
        for variable_index, variable in enumerate(raw["variable"]):
            rows.append(
                {
                    "lead_time_hours": int(lead),
                    "variable": str(variable),
                    "family": search.variable_family(str(variable)),
                    "raw_rmse": float(raw_rmse[lead_index, variable_index]),
                    "scheduled_rmse": float(ttc_rmse[lead_index, variable_index]),
                    "relative_rmse_gain": float(gain[lead_index, variable_index]),
                    "raw_acc": float(raw_acc[lead_index, variable_index]),
                    "scheduled_acc": float(ttc_acc[lead_index, variable_index]),
                    "acc_delta": float(delta[lead_index, variable_index]),
                }
            )
    by_lead = {}
    for lead_index, lead in enumerate(raw["lead_time"]):
        by_lead[str(int(lead))] = {
            "mean_relative_rmse_gain": float(gain[lead_index].mean()),
            "negative_rmse_cells": int((gain[lead_index] < 0).sum()),
            "mean_acc_delta": float(delta[lead_index].mean()),
            "negative_acc_cells": int((delta[lead_index] < 0).sum()),
        }
    summary = search.summarize(raw, calibrated, 0, len(raw["init_time"]))
    summary["by_lead"] = by_lead
    return summary, rows


def target_pass(summary: dict[str, Any], target: dict[str, Any]) -> bool:
    return bool(
        summary["mean_relative_rmse_gain"] >= target["minimum_mean_relative_rmse_gain"]
        and summary["negative_rmse_cell_fraction"] <= target["maximum_negative_rmse_cell_fraction"]
        and summary["mean_acc_delta"] >= target["minimum_mean_acc_delta"]
    )


def render_report(
    primary_summary: dict[str, Any],
    secondary_summary: dict[str, Any],
    primary: dict[str, Any],
    schedule: dict[str, Any],
) -> None:
    target = json.loads(search.DESIGN.read_text())["prospective_success"]
    primary_pass = target_pass(primary_summary, target)
    secondary_pass = target_pass(secondary_summary, target)
    result = {
        "prospective_success_target": target,
        "primary_global": {
            **primary_summary,
            "selected_candidate_id": primary["selected_candidate_id"],
            "prospective_success_pass": primary_pass,
            "frozen_at": primary["created_at"],
        },
        "secondary_family_lead_schedule": {
            **secondary_summary,
            "prospective_success_pass": secondary_pass,
            "frozen_at": schedule["created_at"],
        },
    }
    (ROOT / "PROSPECTIVE_2020_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Prospective 2020 GraphCast + SphereTTC",
        "",
        "## Primary: one global parameter set",
        "",
        f"- Frozen candidate: `{primary['selected_candidate_id']}`",
        f"- Result: {'PASS' if primary_pass else 'FAIL'}",
        f"- Mean cell-relative RMSE gain: {100 * primary_summary['mean_relative_rmse_gain']:.4f}%",
        f"- RMSE-regressed cells: {primary_summary['negative_rmse_cells']}/245",
        f"- Macro ACC change: {primary_summary['mean_acc_delta']:+.8f}",
        f"- ACC-regressed cells: {primary_summary['negative_acc_cells']}/245",
        "",
        "| Lead | Mean RMSE gain | RMSE regressions | Mean ACC change | ACC regressions |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lead, row in primary_summary["by_lead"].items():
        lines.append(
            f"| {lead}h | {100 * row['mean_relative_rmse_gain']:.4f}% | "
            f"{row['negative_rmse_cells']}/49 | {row['mean_acc_delta']:+.8f} | "
            f"{row['negative_acc_cells']}/49 |"
        )
    lines.extend(["", "## Secondary: family/lead parameter schedule", ""])
    lines.extend(
        [
            f"- Result: {'PASS' if secondary_pass else 'FAIL'}",
            f"- Mean cell-relative RMSE gain: {100 * secondary_summary['mean_relative_rmse_gain']:.4f}%",
            f"- RMSE-regressed cells: {secondary_summary['negative_rmse_cells']}/245",
            f"- Macro ACC change: {secondary_summary['mean_acc_delta']:+.8f}",
            f"- ACC-regressed cells: {secondary_summary['negative_acc_cells']}/245",
            "",
            "| Lead | Mean RMSE gain | RMSE regressions | Mean ACC change | ACC regressions |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for lead, row in secondary_summary["by_lead"].items():
        lines.append(
            f"| {lead}h | {100 * row['mean_relative_rmse_gain']:.4f}% | "
            f"{row['negative_rmse_cells']}/49 | {row['mean_acc_delta']:+.8f} | "
            f"{row['negative_acc_cells']}/49 |"
        )
    lines.extend(
        [
            "",
            "Both settings were frozen from 2017 before any 2020 forecast-error",
            "metric was computed or inspected. Both 2019 warmup and 2020 forecasts",
            "were generated from the official local checkpoint on the same pair",
            "of A100 GPUs.",
        ]
    )
    (ROOT / "PROSPECTIVE_2020_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    schedule = json.loads((ROOT / "FROZEN_SCHEDULE_2017.json").read_text())
    primary = json.loads((ROOT / "FROZEN_PRIMARY_GLOBAL_2017.json").read_text())
    validate_inputs(schedule, primary)
    output_root = ROOT / "metrics/holdout_2020"
    raw_path = output_root / "raw.npz"
    if not args.score_only:
        execute(
            "holdout_2020_raw",
            base_command(raw_path, ROOT / "profiles/holdout_2020_raw.json") + ["--method", "raw"],
            0,
        )
        tasks = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            candidate_ids = sorted(
                set(schedule["selected_candidate_ids"]) | {primary["selected_candidate_id"]}
            )
            for index, candidate_id in enumerate(candidate_ids):
                command = base_command(
                    output_root / f"{candidate_id}.npz",
                    ROOT / f"profiles/holdout_2020_{candidate_id}.json",
                ) + [
                    "--method",
                    "sphere_ttc",
                    "--params-json",
                    str(ROOT / f"configs/generated/{candidate_id}.json"),
                    "--warmup-cache",
                    str(WARMUP),
                    "--trim-warmup-cache",
                    "--warmup-truth-root",
                    str(WARMUP_TRUTH),
                ]
                tasks.append(
                    executor.submit(execute, f"holdout_2020_{candidate_id}", command, index % 2)
                )
            for future in as_completed(tasks):
                future.result()
    raw = search.load(raw_path)
    candidate_ids = sorted(
        set(schedule["selected_candidate_ids"]) | {primary["selected_candidate_id"]}
    )
    metrics = {
        candidate_id: search.load(output_root / f"{candidate_id}.npz")
        for candidate_id in candidate_ids
    }
    secondary = search.combine_schedule(raw, metrics, schedule["schedule"])
    variables = [str(value) for value in raw["variable"]]
    families = sorted({search.variable_family(variable) for variable in variables})
    primary_schedule = {
        str(int(lead)): {family: primary["selected_candidate_id"] for family in families}
        for lead in raw["lead_time"]
    }
    primary_schedule_payload = {"schedule": primary_schedule}
    primary_result = search.combine_schedule(raw, metrics, primary_schedule)
    save_composite(primary_result, primary_schedule_payload, "primary_global_sphere_ttc.npz")
    save_composite(secondary, schedule, "scheduled_sphere_ttc.npz")
    primary_summary, primary_rows = cell_details(raw, primary_result)
    secondary_summary, secondary_rows = cell_details(raw, secondary)
    with (ROOT / "PROSPECTIVE_2020_PRIMARY_CELLS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary_rows[0]))
        writer.writeheader()
        writer.writerows(primary_rows)
    with (ROOT / "PROSPECTIVE_2020_SCHEDULED_CELLS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(secondary_rows[0]))
        writer.writeheader()
        writer.writerows(secondary_rows)
    render_report(primary_summary, secondary_summary, primary, schedule)
    print(
        "PROSPECTIVE_2020_COMPLETE",
        f"primary_rmse_gain={100 * primary_summary['mean_relative_rmse_gain']:.4f}%",
        f"primary_acc_delta={primary_summary['mean_acc_delta']:+.8f}",
        f"secondary_rmse_gain={100 * secondary_summary['mean_relative_rmse_gain']:.4f}%",
        f"secondary_acc_delta={secondary_summary['mean_acc_delta']:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
