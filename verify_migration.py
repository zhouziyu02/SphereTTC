#!/usr/bin/env python3
"""Read-only integrity and reproducibility verifier for the SphereTTC repository.

Checks the scoped package (excluding local paper/, .venv/, and runs/):
every manifest digest; frozen input hashes; the lightweight data policy;
5390 rows / 11 backbones / 22 configurations / 49 variables / 5 leads; exact
regeneration of all current tables and results/; frozen GraphCast summaries; and
AST equivalence of the retained SphereTTC calibration and memory core.
No forecasting tensors, weights, GPU, network, or file writes are required.
"""
from __future__ import annotations

import ast
import copy
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
from scripts.tables.build_latest_results import latest_csv_text, render_latest, render_summary
from scripts.tables.render_main_results_49_variables_zh import MODEL_ORDER, _load_rows, _render

BASELINE = ROOT / "artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv"
ORIGINAL_TABLES = BASELINE.with_name("MAIN_RESULTS_49_VARIABLES_ZH.md")
LATEST_ROOT = ROOT / "artifacts/graphcast_spherettc_20260804"
LATEST = LATEST_ROOT / "LATEST_RESULTS.csv"
LATEST_TABLES = LATEST_ROOT / "LATEST_RESULTS_49_VARIABLES_ZH.md"
COMPARISON = LATEST_ROOT / "COMPARISON_WITH_FROZEN_49_TABLES.json"
MANIFEST = ROOT / "MANIFEST.sha256"
CONSOLIDATION_ARCHIVE = "archive/pre_consolidation_20261003"
CONSOLIDATED_SYMBOLS = (
    ("SphereTTCConfig", "src/ttc/sphere.py"),
    ("_complex_delta_clamp", "src/ttc/sphere.py"),
    ("SphereTTCCalibrator", "src/ttc/sphere.py"),
    ("EligibleMemory", "src/ttc/memory.py"),
    ("assert_no_future_targets", "src/ttc/memory.py"),
)
# Independently pinned before cleanup: the SphereTTC result subset and originals.
FROZEN_HASHES = {
    "artifacts/shared/s2s_daily_54var_stats.json": "892409c537a7f3c624ac4f81681ae64fce853f10c84a4e9774838cfd95969227",
    "artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv": "ef47e6683317998ec82734c5fd3ae45a8c1422f03484b244ef0aa4fb05781e5e",
    "artifacts/graphcast_spherettc_20260804/GRAPHCAST_L64_S08_2018_2019.csv": "cc8cd963efe38792fe3c4616ae2705686771b72ae54505fc15be579489e9d080",
    "artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv": "0095fa0699856df285896e8a126a601f5bede076676f47eb68a9a83f50b9dc50",
    "archive/pre_consolidation_20261003/src/ttc/sphere.py": "5b153a4e6cbdff926bf227b093d614425c56559975381f7fa1c1bab84120faac",
    "archive/pre_consolidation_20261003/src/ttc/memory.py": "4c4c6af4c56fffddb0f59e2e01ef778491455985fa05612e0dcfd3f30d5a9430",
}
KEY = ("model", "method", "period", "lead_time_hours", "variable")
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
LOCAL_DIRECTORIES = {"paper", ".venv", "runs"}
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".h5", ".hdf5", ".npy", ".npz", ".pickle", ".pkl", ".pt", ".pth", ".safetensors", ".tar", ".tgz", ".zip"}
FORBIDDEN_PARTS = {"cache", "caches", "checkpoint", "checkpoints", "data", "datasets", "external", "metrics", "packages"}


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


def repository_files() -> set[str]:
    return {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
            if p.is_file() and p.relative_to(ROOT).parts[0] not in LOCAL_DIRECTORIES
            and not (set(p.relative_to(ROOT).parts) & IGNORED_PARTS)
            and p.name != ".DS_Store" and p != MANIFEST}


def verify_manifest() -> int:
    expected = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        relative = relative.removeprefix("./")
        require(relative not in expected, f"duplicate manifest path: {relative}")
        expected[relative] = digest
    actual_files = repository_files()
    require(set(expected) == actual_files,
            f"manifest membership mismatch: missing={sorted(set(expected) - actual_files)} extra={sorted(actual_files - set(expected))}")
    for relative, digest in expected.items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest,
                f"manifest digest mismatch: {relative}")
    for relative, digest in FROZEN_HASHES.items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest,
                f"frozen evidence changed: {relative}")
    return len(expected)


def verify_lightweight_policy() -> int:
    files = repository_files()
    for relative in files:
        path = Path(relative)
        require(not ({part.lower() for part in path.parts} & FORBIDDEN_PARTS),
                f"raw-data/weight directory is not allowed: {relative}")
        require(not ({suffix.lower() for suffix in path.suffixes} & FORBIDDEN_SUFFIXES),
                f"raw-data/weight/archive file is not allowed: {relative}")
        require(".tar.part-" not in path.name, f"split archive is not allowed: {relative}")
    return len(files)


def verify_tables_and_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    baseline_fields, baseline = read_csv(BASELINE)
    latest_fields, latest = read_csv(LATEST)
    require(baseline_fields == latest_fields, "CSV schema mismatch")
    expected_configs = {(model, method) for model in MODEL_ORDER for method in ("raw", "sphere_ttc")}
    for label, rows in (("baseline", baseline), ("latest", latest)):
        require(len(rows) == 5390, f"{label} row count is not 5,390")
        require({row["model"] for row in rows} == set(MODEL_ORDER), f"{label} backbones differ")
        require({(row["model"], row["method"]) for row in rows} == expected_configs,
                f"{label} does not contain the 22 expected configurations")
        require({row["n_initializations"] for row in rows} == {"730"}, f"{label} initialization count changed")
    original_map, latest_map = keyed(baseline), keyed(latest)
    require(set(original_map) == set(latest_map), "latest result coordinates changed")
    changed = []
    for coordinate, before in original_map.items():
        after = latest_map[coordinate]
        if before != after:
            changed.append(coordinate)
            require(coordinate[:2] == ("graphcast", "sphere_ttc"), f"non-GraphCast result changed: {coordinate}")
            for field in baseline_fields:
                if field not in {"rmse", "acc"}:
                    require(before[field] == after[field], f"metadata changed: {coordinate}/{field}")
    require(len(changed) == 245, f"expected 245 GraphCast replacements, found {len(changed)}")
    require(LATEST.read_bytes() == latest_csv_text().encode("utf-8"), "latest CSV is not byte reproducible")
    variables, units, values = _load_rows([BASELINE])
    rendered = _render([BASELINE.relative_to(ROOT)], variables, units, values)
    require(rendered == ORIGINAL_TABLES.read_text(encoding="utf-8"), "publication 49 tables are not byte reproducible")
    variables, _, _ = _load_rows([LATEST])
    require(len(variables) == 49, "latest CSV does not contain 49 variables")
    rendered = render_latest(LATEST)
    require(rendered == LATEST_TABLES.read_text(encoding="utf-8"), "latest 49 tables are not byte reproducible")
    require(rendered.count('<a id="var-') == 49, "latest Markdown does not contain 49 tables")
    fields, graphcast = read_csv(LATEST_ROOT / "GRAPHCAST_L64_S08_2018_2019.csv")
    require(fields == latest_fields, "GraphCast extract schema mismatch")
    require(graphcast == [row for row in latest if row["model"] == "graphcast"], "GraphCast extract mismatch")
    return baseline, latest


def verify_experiment_summaries(baseline: list[dict[str, str]], latest: list[dict[str, str]]) -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    cases = [
        (summarize_model(baseline, "graphcast", "sphere_ttc"), "frozen_49_table_graphcast_spherettc_2018_2019"),
        (summarize_model(latest, "graphcast", "sphere_ttc"), "latest_global_l64_s08_2018_2019"),
        (summarize_cells(LATEST_ROOT / "parameter_only/test_cells.csv"), "parameter_only_l24_strength050_2018_2019"),
        (summarize_cells(LATEST_ROOT / "schedule/PROSPECTIVE_2020_PRIMARY_CELLS.csv"), "latest_global_l64_s08_2020_prospective"),
        (summarize_cells(LATEST_ROOT / "schedule/PROSPECTIVE_2020_SCHEDULED_CELLS.csv"), "secondary_family_lead_schedule_2020_prospective"),
    ]
    for actual, key in cases:
        compare_summary(actual, comparison[key], key)
    expected = render_summary(comparison)
    require((LATEST_ROOT / "LATEST_EXPERIMENT_SUMMARY_ZH.md").read_text(encoding="utf-8") == expected,
            "GraphCast narrative summary is not byte reproducible")


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


def _normalized_definition(node: ast.AST) -> str:
    """AST dump of a definition with docstrings removed and its own name normalized."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            child.body = body[1:] or [ast.Pass()]
    if hasattr(node, "name"):
        node.name = "_"
    return ast.dump(node, include_attributes=False)


def _top_level_definitions(source: str) -> dict[str, ast.AST]:
    definitions: dict[str, ast.AST] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            definitions[node.targets[0].id] = node
    return definitions


def verify_consolidation() -> int:
    """Compare the retained calibration/memory definitions against pinned originals."""
    canonical = _top_level_definitions((ROOT / "src/spherettc.py").read_text(encoding="utf-8"))
    originals = {}
    for name, original_file in CONSOLIDATED_SYMBOLS:
        if original_file not in originals:
            originals[original_file] = _top_level_definitions(
                (ROOT / CONSOLIDATION_ARCHIVE / original_file).read_text(encoding="utf-8"))
        require(name in canonical, f"src/spherettc.py lacks {name}")
        require(_normalized_definition(canonical[name]) == _normalized_definition(originals[original_file][name]),
                f"src/spherettc.py::{name} differs from {original_file}::{name}")
    return len(CONSOLIDATED_SYMBOLS)


def verify_results_summary() -> int:
    from tools.build_results_summary import OUT, build
    outputs = build()
    for name, text in outputs.items():
        require((OUT / name).read_text(encoding="utf-8") == text,
                f"results/{name} is not byte reproducible; run tools/build_results_summary.py")
    spherettc = json.loads(outputs["SOTA_SUMMARY.json"])["spherettc_final_version"]
    require(spherettc["backbones_improved_macro_nrmse"] == spherettc["backbones"] == 11,
            "SphereTTC no longer improves macro nRMSE on all 11 backbones")
    require(spherettc["graphcast_l64_s08"]["prospective_2020"]["rmse_regressed_cells"] == 0,
            "GraphCast prospective RMSE regression count changed")
    return len(outputs)


def main() -> None:
    manifest_files = verify_manifest()
    lightweight_files = verify_lightweight_policy()
    baseline, latest = verify_tables_and_rows()
    verify_experiment_summaries(baseline, latest)
    summaries = verify_results_summary()
    consolidated = verify_consolidation()
    print("SPHERETTC_REPRODUCIBLE "
          f"manifest_files={manifest_files} lightweight_files={lightweight_files} "
          "baseline_rows=5390 latest_rows=5390 backbones=11 configurations=22 "
          "variables=49 tables=49 graphcast_replacements=245 raw_data=false weights=0 "
          f"results_summaries={summaries} spherettc_backbones_improved=11/11 "
          f"consolidated_symbols_identical={consolidated}")


if __name__ == "__main__":
    main()
