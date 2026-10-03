#!/usr/bin/env python3
"""Read-only verifier for the portable SOON experiment migration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_latest_results import render_latest
from scripts.render_main_results_49_variables_zh import _load_rows, _render


BASELINE = ROOT / "artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv"
SPHEREDYN = ROOT / (
    "artifacts/spheredyn_spherettc_open_goal_20260801/"
    "main_prediction_v2_seed44/paper_metrics/SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv"
)
ORIGINAL_TABLES = ROOT / (
    "artifacts/spheredyn_spherettc_open_goal_20260801/"
    "MAIN_RESULTS_49_VARIABLES_ZH.md"
)
LATEST_ROOT = ROOT / "artifacts/graphcast_spherettc_20260804"
LATEST = LATEST_ROOT / "LATEST_RESULTS.csv"
LATEST_TABLES = LATEST_ROOT / "LATEST_RESULTS_49_VARIABLES_ZH.md"
COMPARISON = LATEST_ROOT / "COMPARISON_WITH_FROZEN_49_TABLES.json"
MANIFEST = ROOT / "MIGRATION_MANIFEST.sha256"

ALLOWED_WEIGHTS = {
    (
        "artifacts/spheredyn_spherettc_open_goal_20260801/"
        "spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/"
        "checkpoints/spheredyn_v9_multiscale.pt"
    ): {
        "bytes": 16_302_127,
        "sha256": "121f9b900db554f175f121e629ac64a10ce2697e7cf4bfd224f09984c6bbf453",
    },
    (
        "artifacts/spheredyn_spherettc_open_goal_20260801/"
        "main_prediction_v2_seed44/final/FITTED_WEIGHTS_2017.npz"
    ): {
        "bytes": 35_588_308,
        "sha256": "5f3bb4785581768eab9bc5576068073d4ea9b21da36c49053b0ab4dd807fc397",
    },
}

KEY = ("model", "method", "period", "lead_time_hours", "variable")
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".zip",
}
FORBIDDEN_PARTS = {
    "cache",
    "caches",
    "checkpoint",
    "checkpoints",
    "data",
    "datasets",
    "external",
    "metrics",
    "packages",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        coordinate = tuple(row[name] for name in KEY)
        require(coordinate not in result, f"duplicate row: {coordinate}")
        result[coordinate] = row
    return result


def verify_manifest() -> int:
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        expected[relative] = digest
    for relative, digest in expected.items():
        path = ROOT / relative
        require(path.is_file(), f"manifest file missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == digest, f"manifest digest mismatch: {relative}")
    return len(expected)


def verify_lightweight_policy() -> int:
    checked = 0
    verified_weights: set[str] = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts):
            continue
        relative_text = relative.as_posix()
        if relative_text in ALLOWED_WEIGHTS:
            expected = ALLOWED_WEIGHTS[relative_text]
            require(path.is_file(), f"allowed weight is not a file: {relative}")
            require(path.stat().st_size == expected["bytes"], f"weight size mismatch: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == expected["sha256"], f"weight digest mismatch: {relative}")
            verified_weights.add(relative_text)
            checked += 1
            continue
        if path.is_dir() and any(
            weight.startswith(relative_text + "/") for weight in ALLOWED_WEIGHTS
        ):
            continue
        lower_parts = {part.lower() for part in relative.parts}
        require(
            not (lower_parts & FORBIDDEN_PARTS),
            f"raw-data/weight directory is not allowed: {relative}",
        )
        if path.is_file():
            suffixes = {suffix.lower() for suffix in path.suffixes}
            require(
                not (suffixes & FORBIDDEN_SUFFIXES),
                f"raw-data/weight/archive file is not allowed: {relative}",
            )
            require(".tar.part-" not in path.name, f"split archive is not allowed: {relative}")
            checked += 1
    require(verified_weights == set(ALLOWED_WEIGHTS), "one or more required weights are missing")
    return checked


def close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=atol):
        raise RuntimeError(f"{label}: actual={actual!r}, expected={expected!r}")


def summarize_values(gain: np.ndarray, acc_delta: np.ndarray, leads: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cell_count": int(gain.size),
        "mean_relative_rmse_gain": float(gain.mean()),
        "median_relative_rmse_gain": float(np.median(gain)),
        "minimum_relative_rmse_gain": float(gain.min()),
        "negative_rmse_cells": int((gain < 0).sum()),
        "negative_rmse_cell_fraction": float((gain < 0).mean()),
        "mean_acc_delta": float(acc_delta.mean()),
        "minimum_acc_delta": float(acc_delta.min()),
        "negative_acc_cells": int((acc_delta < 0).sum()),
        "by_lead": {},
    }
    for lead in sorted(set(int(value) for value in leads)):
        selected = leads == lead
        result["by_lead"][str(lead)] = {
            "mean_relative_rmse_gain": float(gain[selected].mean()),
            "negative_rmse_cells": int((gain[selected] < 0).sum()),
            "mean_acc_delta": float(acc_delta[selected].mean()),
            "negative_acc_cells": int((acc_delta[selected] < 0).sum()),
        }
    return result


def summarize_model(rows: list[dict[str, str]], model: str, method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    raw = {
        (row["lead_time_hours"], row["variable"]): row
        for row in selected
        if row["method"] == "raw"
    }
    calibrated = {
        (row["lead_time_hours"], row["variable"]): row
        for row in selected
        if row["method"] == method
    }
    require(len(raw) == len(calibrated) == 245, f"incomplete {model}/{method} result")
    coordinates = sorted(raw, key=lambda item: (int(item[0]), item[1]))
    gains = np.asarray(
        [1.0 - float(calibrated[key]["rmse"]) / float(raw[key]["rmse"]) for key in coordinates]
    )
    deltas = np.asarray(
        [float(calibrated[key]["acc"]) - float(raw[key]["acc"]) for key in coordinates]
    )
    leads = np.asarray([int(key[0]) for key in coordinates])
    return summarize_values(gains, deltas, leads)


def summarize_cells(path: Path) -> dict[str, Any]:
    _, rows = read_csv(path)
    require(len(rows) == 245, f"expected 245 cells: {path.relative_to(ROOT)}")
    gains = np.asarray([float(row["relative_rmse_gain"]) for row in rows])
    deltas = np.asarray([float(row["acc_delta"]) for row in rows])
    leads = np.asarray([int(row["lead_time_hours"]) for row in rows])
    return summarize_values(gains, deltas, leads)


def compare_summary(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    numeric = (
        "mean_relative_rmse_gain",
        "median_relative_rmse_gain",
        "minimum_relative_rmse_gain",
        "negative_rmse_cell_fraction",
        "mean_acc_delta",
        "minimum_acc_delta",
    )
    integer = ("negative_rmse_cells", "negative_acc_cells")
    for name in numeric:
        if name in expected:
            close(float(actual[name]), float(expected[name]), f"{label}.{name}")
    for name in integer:
        if name in expected:
            require(int(actual[name]) == int(expected[name]), f"{label}.{name} mismatch")
    for lead, expected_lead in expected.get("by_lead", {}).items():
        actual_lead = actual["by_lead"][lead]
        for name in ("mean_relative_rmse_gain", "mean_acc_delta"):
            close(float(actual_lead[name]), float(expected_lead[name]), f"{label}.{lead}.{name}")
        for name in ("negative_rmse_cells", "negative_acc_cells"):
            require(
                int(actual_lead[name]) == int(expected_lead[name]),
                f"{label}.{lead}.{name} mismatch",
            )


def verify_tables_and_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    baseline_fields, baseline = read_csv(BASELINE)
    sphere_fields, sphere = read_csv(SPHEREDYN)
    latest_fields, latest = read_csv(LATEST)
    require(baseline_fields == sphere_fields == latest_fields, "CSV schema mismatch")
    require(len(baseline) == 5390, "baseline row count is not 5,390")
    require(len(sphere) == 490, "SphereDyn row count is not 490")
    require(len(latest) == 5880, "latest row count is not 5,880")

    original = baseline + sphere
    original_map = keyed(original)
    latest_map = keyed(latest)
    require(set(original_map) == set(latest_map), "latest result coordinates changed")
    changed: list[tuple[str, ...]] = []
    for coordinate in original_map:
        before = original_map[coordinate]
        after = latest_map[coordinate]
        if before != after:
            changed.append(coordinate)
            require(
                coordinate[0] == "graphcast" and coordinate[1] == "sphere_ttc",
                f"non-GraphCast result changed: {coordinate}",
            )
            for field in baseline_fields:
                if field not in {"rmse", "acc"}:
                    require(before[field] == after[field], f"metadata changed: {coordinate}/{field}")
    require(len(changed) == 245, f"expected 245 GraphCast replacements, found {len(changed)}")

    variables, units, values = _load_rows([BASELINE, SPHEREDYN])
    rendered_original = _render(
        [BASELINE.relative_to(ROOT), SPHEREDYN.relative_to(ROOT)], variables, units, values
    )
    require(rendered_original == ORIGINAL_TABLES.read_text(encoding="utf-8"), "original 49 tables are not byte reproducible")

    variables, _, _ = _load_rows([LATEST])
    require(len(variables) == 49, "latest CSV does not contain 49 variables")
    rendered_latest = render_latest(LATEST)
    require(rendered_latest == LATEST_TABLES.read_text(encoding="utf-8"), "latest 49 tables are not byte reproducible")
    require(rendered_latest.count('<a id="var-') == 49, "latest Markdown does not contain 49 tables")

    graphcast_fields, graphcast_rows = read_csv(LATEST_ROOT / "GRAPHCAST_L64_S08_2018_2019.csv")
    require(graphcast_fields == latest_fields, "GraphCast extract schema mismatch")
    require(graphcast_rows == [row for row in latest if row["model"] == "graphcast"], "GraphCast extract mismatch")
    return baseline, latest


def verify_experiment_summaries(baseline: list[dict[str, str]], latest: list[dict[str, str]]) -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    compare_summary(
        summarize_model(baseline, "graphcast", "sphere_ttc"),
        comparison["frozen_49_table_graphcast_spherettc_2018_2019"],
        "frozen_graphcast",
    )
    compare_summary(
        summarize_model(latest, "graphcast", "sphere_ttc"),
        comparison["latest_global_l64_s08_2018_2019"],
        "latest_graphcast",
    )
    compare_summary(
        summarize_cells(LATEST_ROOT / "parameter_only/test_cells.csv"),
        comparison["parameter_only_l24_strength050_2018_2019"],
        "parameter_only",
    )
    compare_summary(
        summarize_cells(LATEST_ROOT / "schedule/PROSPECTIVE_2020_PRIMARY_CELLS.csv"),
        comparison["latest_global_l64_s08_2020_prospective"],
        "prospective_primary",
    )
    compare_summary(
        summarize_cells(LATEST_ROOT / "schedule/PROSPECTIVE_2020_SCHEDULED_CELLS.csv"),
        comparison["secondary_family_lead_schedule_2020_prospective"],
        "prospective_schedule",
    )

    gate = json.loads(
        (ROOT / "artifacts/spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/MAIN_GOAL_GATE.json").read_text(encoding="utf-8")
    )
    require(gate["status"] == "MAIN_PREDICTION_GOAL_ACHIEVED", "SphereDyn main goal is not achieved")
    close(float(gate["gain_percent"]), 25.282903110901678, "SphereDyn gain percent")


def main() -> None:
    manifest_files = verify_manifest()
    lightweight_files = verify_lightweight_policy()
    baseline, latest = verify_tables_and_rows()
    verify_experiment_summaries(baseline, latest)
    print(
        "SOON_MIGRATION_REPRODUCIBLE "
        f"manifest_files={manifest_files} lightweight_files={lightweight_files} "
        "baseline_rows=5390 spheredyn_rows=490 latest_rows=5880 "
        "variables=49 tables=49 graphcast_replacements=245 raw_data=false "
        "allowlisted_weights=2 weight_bytes=51890435"
    )


if __name__ == "__main__":
    main()
