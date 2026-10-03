#!/usr/bin/env python
"""Read-only verification of the frozen 49 tables and later GraphCast results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import build_latest_results as latest


ARTIFACT = REPOSITORY / "artifacts/graphcast_spherettc_20260804"
PARAMETER = ARTIFACT / "parameter_only"
SCHEDULE = ARTIFACT / "schedule"


def import_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_close(actual: Any, expected: Any, label: str, tolerance: float = 1e-12) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeError(f"{label}: expected mapping")
        for key, value in expected.items():
            if key not in actual:
                raise RuntimeError(f"{label}: missing key {key}")
            assert_close(actual[key], value, f"{label}.{key}", tolerance)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{label}: list mismatch")
        for index, value in enumerate(expected):
            assert_close(actual[index], value, f"{label}[{index}]", tolerance)
        return
    if isinstance(expected, (float, np.floating)):
        if not np.isclose(float(actual), float(expected), rtol=tolerance, atol=tolerance):
            raise RuntimeError(f"{label}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_original_49_tables() -> str:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_main_results_reproducibility.py"],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout.strip()
    if "MAIN_RESULTS_REPRODUCIBLE" not in output:
        raise RuntimeError("original main verifier did not report success")
    return output


def verify_parameter_only() -> dict[str, Any]:
    runner = import_file("portable_parameter_runner", PARAMETER / "tools/run_experiment.py")
    design = runner.load_design()
    candidates = runner.candidate_params(design)
    raw_validation = runner.load_metric(PARAMETER / "metrics/validation/raw.npz")
    scored = []
    for candidate in candidates:
        candidate_id = candidate["id"]
        calibrated = runner.load_metric(
            PARAMETER / f"metrics/validation/{candidate_id}.npz"
        )
        summary, _ = runner.paired_summary(
            raw_validation,
            calibrated,
            int(design["score_after_initialization"]),
        )
        guards = design["validation_guards"]
        eligible = bool(
            summary["negative_rmse_cell_fraction"]
            <= guards["maximum_negative_rmse_cell_fraction"]
            and summary["mean_acc_delta"] >= guards["minimum_macro_acc_delta"]
        )
        scored.append({"id": candidate_id, "eligible": eligible, **summary})
    eligible = [item for item in scored if item["eligible"]]
    pool = eligible if eligible else scored
    best = max(
        pool,
        key=lambda item: (
            item["mean_relative_rmse_gain"],
            item["mean_acc_delta"],
            -item["negative_rmse_cell_fraction"],
        ),
    )
    frozen_selection = json.loads((PARAMETER / "selection.json").read_text())
    if best["id"] != frozen_selection["best_id"]:
        raise RuntimeError("parameter-only 2017 selection is not reproducible")
    if (not eligible) != frozen_selection["fallback_used"]:
        raise RuntimeError("parameter-only fallback decision is not reproducible")

    raw_test = runner.concatenate(
        [PARAMETER / "metrics/test/raw_2018.npz", PARAMETER / "metrics/test/raw_2019.npz"]
    )
    calibrated_test = runner.concatenate(
        [
            PARAMETER / "metrics/test/sphere_ttc_2018.npz",
            PARAMETER / "metrics/test/sphere_ttc_2019.npz",
        ]
    )
    computed, details = runner.paired_summary(raw_test, calibrated_test, 0)
    frozen = json.loads((PARAMETER / "test_summary.json").read_text())
    for key in computed:
        assert_close(computed[key], frozen[key], f"parameter_test.{key}")
    if len(details) != 245 or len(csv_rows(PARAMETER / "test_cells.csv")) != 245:
        raise RuntimeError("parameter-only cell table is incomplete")
    if frozen["primary_expected_effect_pass"] or frozen["strict_acc_pass"]:
        raise RuntimeError("parameter-only frozen FAIL decision changed")

    canonical_raw = latest.load_metric(
        latest.ORIGINAL_GRAPHCAST_METRICS / "raw/per_initialization_metrics.npz"
    )
    for key in (
        "init_time",
        "lead_time",
        "variable",
        "mse",
        "mae",
        "bias",
        "acc_numerator",
        "acc_prediction_energy",
        "acc_target_energy",
    ):
        if not np.array_equal(raw_test[key], canonical_raw[key]):
            raise RuntimeError(f"parameter-only raw reproduction differs for {key}")
    return frozen


def recompute_schedule_mapping(
    search: ModuleType,
    items: list[dict[str, Any]],
    raw: dict[str, np.ndarray],
    metrics: dict[str, dict[str, np.ndarray]],
    design: dict[str, Any],
) -> dict[str, dict[str, str]]:
    start, stop = design["search_initialization_slice"]
    guards = design["search_eligibility"]
    variables = [str(value) for value in raw["variable"]]
    families = sorted({search.variable_family(variable) for variable in variables})
    indices = {
        family: np.asarray(
            [
                index
                for index, variable in enumerate(variables)
                if search.variable_family(variable) == family
            ]
        )
        for family in families
    }
    cell_metrics = {
        item["id"]: search.window_cell_metrics(raw, metrics[item["id"]], start, stop)
        for item in items
    }
    result: dict[str, dict[str, str]] = {}
    for lead_index, lead in enumerate(raw["lead_time"]):
        lead_key = str(int(lead))
        result[lead_key] = {}
        for family in families:
            family_indices = indices[family]
            best: tuple[float, float, str] | None = None
            for item in items:
                gain, delta = cell_metrics[item["id"]]
                current_gain = gain[lead_index, family_indices]
                current_delta = delta[lead_index, family_indices]
                mean_gain = float(current_gain.mean())
                negative_fraction = float((current_gain < 0).mean())
                mean_delta = float(current_delta.mean())
                if not (
                    mean_gain > guards["minimum_mean_relative_rmse_gain"]
                    and negative_fraction
                    <= guards["maximum_negative_rmse_cell_fraction"]
                    and mean_delta >= guards["minimum_mean_acc_delta"]
                ):
                    continue
                key = (mean_gain, mean_delta, item["id"])
                if best is None or key[:2] > best[:2]:
                    best = key
            result[lead_key][family] = "noop" if best is None else best[2]
    return result


def verify_schedule() -> dict[str, Any]:
    search = import_file("search_schedule", SCHEDULE / "tools/search_schedule.py")
    design = json.loads((SCHEDULE / "configs/search_space.json").read_text())
    items = search.candidates()
    raw_validation = search.load(SCHEDULE / "metrics/validation/raw.npz")
    metrics = {
        item["id"]: search.load(SCHEDULE / f"metrics/validation/{item['id']}.npz")
        for item in items
    }
    search_start, search_stop = design["search_initialization_slice"]
    holdout_start, holdout_stop = design["internal_holdout_slice"]
    rule = design["primary_global_selection"]
    eligible = []
    for item in items:
        first = search.summarize(
            raw_validation, metrics[item["id"]], search_start, search_stop
        )
        second = search.summarize(
            raw_validation, metrics[item["id"]], holdout_start, holdout_stop
        )
        passed = all(
            summary["mean_relative_rmse_gain"]
            >= rule["minimum_mean_relative_rmse_gain_each_window"]
            and summary["negative_rmse_cell_fraction"]
            <= rule["maximum_negative_rmse_cell_fraction_each_window"]
            and summary["mean_acc_delta"]
            >= rule["minimum_mean_acc_delta_each_window"]
            for summary in (first, second)
        )
        if passed:
            eligible.append(
                {
                    "id": item["id"],
                    "first": first,
                    "second": second,
                    "gain": min(
                        first["mean_relative_rmse_gain"],
                        second["mean_relative_rmse_gain"],
                    ),
                    "acc": min(first["mean_acc_delta"], second["mean_acc_delta"]),
                }
            )
    selected = max(eligible, key=lambda item: (item["gain"], item["acc"], item["id"]))
    primary = json.loads((SCHEDULE / "FROZEN_PRIMARY_GLOBAL_2017.json").read_text())
    if selected["id"] != primary["selected_candidate_id"]:
        raise RuntimeError("global 2017 primary selection is not reproducible")
    assert_close(selected["first"], primary["search_summary"], "primary.search")
    assert_close(selected["second"], primary["internal_holdout_summary"], "primary.holdout")

    frozen_schedule = json.loads((SCHEDULE / "FROZEN_SCHEDULE_2017.json").read_text())
    mapping = recompute_schedule_mapping(search, items, raw_validation, metrics, design)
    if mapping != frozen_schedule["schedule"]:
        raise RuntimeError("family/lead schedule is not reproducible from 2017")
    combined_validation = search.combine_schedule(raw_validation, metrics, mapping)
    internal = search.summarize(
        raw_validation, combined_validation, holdout_start, holdout_stop
    )
    assert_close(internal, frozen_schedule["internal_holdout_summary"], "schedule.holdout")

    retrospective_root = SCHEDULE / "metrics/retrospective_2018_2019"
    retrospective_raw = latest.concatenate(
        [retrospective_root / "raw_2018.npz", retrospective_root / "raw_2019.npz"]
    )
    retrospective_ttc = latest.concatenate(
        [
            retrospective_root / "l64_s08_2018.npz",
            retrospective_root / "l64_s08_2019.npz",
        ]
    )
    retrospective = latest.paired_summary(retrospective_raw, retrospective_ttc)
    frozen_retrospective = json.loads(
        (SCHEDULE / "RETROSPECTIVE_2018_2019_PRIMARY.json").read_text()
    )
    for key, value in retrospective.items():
        if key in frozen_retrospective:
            assert_close(value, frozen_retrospective[key], f"retrospective.{key}")

    holdout = SCHEDULE / "metrics/holdout_2020"
    raw_2020 = search.load(holdout / "raw.npz")
    selected_ids = sorted(
        set(frozen_schedule["selected_candidate_ids"])
        | {primary["selected_candidate_id"]}
    )
    holdout_metrics = {
        candidate_id: search.load(holdout / f"{candidate_id}.npz")
        for candidate_id in selected_ids
    }
    secondary = search.combine_schedule(
        raw_2020, holdout_metrics, frozen_schedule["schedule"]
    )
    variables = [str(value) for value in raw_2020["variable"]]
    families = sorted({search.variable_family(variable) for variable in variables})
    primary_mapping = {
        str(int(lead)): {
            family: primary["selected_candidate_id"] for family in families
        }
        for lead in raw_2020["lead_time"]
    }
    primary_result = search.combine_schedule(raw_2020, holdout_metrics, primary_mapping)
    frozen_2020 = json.loads((SCHEDULE / "PROSPECTIVE_2020_SUMMARY.json").read_text())
    computed_primary = latest.paired_summary(raw_2020, primary_result)
    computed_secondary = latest.paired_summary(raw_2020, secondary)
    for label, computed, frozen in (
        ("primary_2020", computed_primary, frozen_2020["primary_global"]),
        ("secondary_2020", computed_secondary, frozen_2020["secondary_family_lead_schedule"]),
    ):
        for key, value in computed.items():
            if key in frozen:
                assert_close(value, frozen[key], f"{label}.{key}")
    if not frozen_2020["primary_global"]["prospective_success_pass"]:
        raise RuntimeError("frozen 2020 primary no longer passes")
    if len(csv_rows(SCHEDULE / "PROSPECTIVE_2020_PRIMARY_CELLS.csv")) != 245:
        raise RuntimeError("2020 primary cell table is incomplete")
    if len(csv_rows(SCHEDULE / "PROSPECTIVE_2020_SCHEDULED_CELLS.csv")) != 245:
        raise RuntimeError("2020 schedule cell table is incomplete")

    for computed, path in (
        (primary_result, holdout / "primary_global_sphere_ttc.npz"),
        (secondary, holdout / "scheduled_sphere_ttc.npz"),
    ):
        frozen = search.load(path)
        for key in latest.METRIC_FIELDS:
            if key in frozen and not np.array_equal(computed[key], frozen[key]):
                raise RuntimeError(f"2020 composite metric mismatch: {path.name}/{key}")

    bootstrap = import_file("portable_bootstrap", SCHEDULE / "tools/bootstrap_prospective.py")
    count = len(raw_2020["init_time"])
    block_count = (count + bootstrap.BLOCK_DAYS - 1) // bootstrap.BLOCK_DAYS
    starts = np.random.default_rng(bootstrap.SEED).integers(
        0, count, size=(bootstrap.REPLICATES, block_count), endpoint=False
    )
    computed_bootstrap = {
        "method": "circular moving-block bootstrap over initialization dates",
        "replicates": bootstrap.REPLICATES,
        "block_days": bootstrap.BLOCK_DAYS,
        "seed": bootstrap.SEED,
        "primary_global": bootstrap.evaluate(raw_2020, primary_result, starts),
        "secondary_family_lead_schedule": bootstrap.evaluate(
            raw_2020, secondary, starts
        ),
    }
    frozen_bootstrap = json.loads(
        (SCHEDULE / "PROSPECTIVE_2020_BOOTSTRAP.json").read_text()
    )
    assert_close(computed_bootstrap, frozen_bootstrap, "bootstrap")
    return frozen_2020


def verify_latest_outputs() -> None:
    filenames = (
        "LATEST_RESULTS.csv",
        "GRAPHCAST_L64_S08_2018_2019.csv",
        "LATEST_RESULTS_49_VARIABLES_ZH.md",
        "COMPARISON_WITH_FROZEN_49_TABLES.json",
        "LATEST_EXPERIMENT_SUMMARY_ZH.md",
    )
    with tempfile.TemporaryDirectory(prefix="soon-latest-repro-") as temporary:
        destination = Path(temporary)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/build_latest_results.py",
                "--output-root",
                str(destination),
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        )
        if "LATEST_RESULTS_BUILT" not in completed.stdout:
            raise RuntimeError("latest-result builder did not report success")
        for filename in filenames:
            if (destination / filename).read_bytes() != (ARTIFACT / filename).read_bytes():
                raise RuntimeError(f"latest output is not byte-reproducible: {filename}")
    if len(csv_rows(ARTIFACT / "LATEST_RESULTS.csv")) != 5880:
        raise RuntimeError("latest consolidated CSV does not contain 5,880 rows")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for current in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = current.relative_to(path).as_posix().encode("utf-8")
        size = current.stat().st_size
        digest.update(relative + b"\0" + str(size).encode("ascii") + b"\0")
        with current.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        count += 1
        total += size
    return digest.hexdigest(), count, total


def verify_manifest(full: bool) -> None:
    manifest = json.loads(
        (ARTIFACT / "LATEST_REPRODUCIBILITY_MANIFEST.json").read_text()
    )
    for relative, expected in manifest["sha256"].items():
        path = REPOSITORY / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"latest manifest mismatch: {relative}")
    if full:
        for relative, expected in manifest["tree_sha256"].items():
            actual_hash, count, size = tree_sha256(REPOSITORY / relative)
            if (
                actual_hash != expected["sha256"]
                or count != expected["file_count"]
                or size != expected["total_bytes"]
            ):
                raise RuntimeError(f"latest tree manifest mismatch: {relative}")


def verify_portable_assets() -> None:
    required = (
        "external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz",
        "external/official_checkpoints/graphcast/stats/mean_by_level.nc",
        "external/official_checkpoints/graphcast/stats/diffs_stddev_by_level.nc",
        "external/official_checkpoints/graphcast/stats/stddev_by_level.nc",
        "artifacts/graphcast_spherettc_20260804/schedule/cache/graphcast_2019_current_a100.zarr/.zmetadata",
        "artifacts/graphcast_spherettc_20260804/schedule/cache/graphcast_2020_current_a100.zarr/.zmetadata",
        "artifacts/graphcast_spherettc_20260804/schedule/data/S2S/HOLDOUT_MANIFEST.json",
    )
    missing = [relative for relative in required if not (REPOSITORY / relative).is_file()]
    if missing:
        raise RuntimeError(f"portable assets are missing: {missing}")
    for year, count in ((2019, 206), (2020, 366)):
        profile = json.loads(
            (SCHEDULE / f"profiles/graphcast_{year}_current_a100.json").read_text()
        )
        if not profile["complete"] or profile["shape"][0] != count:
            raise RuntimeError(f"GraphCast {year} cache profile is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-integrity",
        action="store_true",
        help="Also stream every file in the 14 GB migrated experiment bundle.",
    )
    args = parser.parse_args()
    original = verify_original_49_tables()
    parameter = verify_parameter_only()
    prospective = verify_schedule()
    verify_latest_outputs()
    verify_manifest(args.full_integrity)
    verify_portable_assets()
    print(original)
    print(
        "LATEST_RESULTS_REPRODUCIBLE "
        "rows=5880 variables=49 tables=49 "
        f"parameter_only_primary_pass={str(parameter['primary_expected_effect_pass']).lower()} "
        f"global_2020_primary_pass={str(prospective['primary_global']['prospective_success_pass']).lower()} "
        f"full_integrity={str(args.full_integrity).lower()}"
    )


if __name__ == "__main__":
    main()
