#!/usr/bin/env python3
"""One-shot consolidation of SphereTTC and SphereDyn into single files (2026-10-03).

Before running, the new canonical files src/spherettc.py and src/spheredyn.py and
the new tests must already be in place.  This script then

  1. moves the superseded originals byte-identically to
     archive/pre_consolidation_20261003/<same relative path>;
  2. regenerates src/baselines/models.py (baselines only) and
     scripts/spheredyn/evaluate_goal_reference_ensemble.py (I/O helpers only)
     from the archived originals, deterministically;
  3. applies exact, counted, reversible import edits;
  4. writes archive/provenance/CONSOLIDATION_LOG_20261003.json.

verify_migration.py replays both provenance logs to prove that every file of the
2026-08-06 package is still recoverable byte-for-byte, and checks that the code
moved into src/spherettc.py is AST-identical to the archived originals.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "archive/pre_consolidation_20261003"
LOG = ROOT / "archive/provenance/CONSOLIDATION_LOG_20261003.json"

REQUIRED_NEW = [
    "src/spherettc.py",
    "src/spheredyn.py",
    "tests/test_spheredyn.py",
    "tests/test_consolidation_equivalence.py",
]
MOVED = [
    "src/ttc/sphere.py",
    "src/ttc/memory.py",
    "src/baselines/spheredyn_v2.py",
    "src/baselines/spheredyn_v4.py",
    "src/baselines/spheredyn_v6.py",
    "src/baselines/spheredyn_v7.py",
    "src/baselines/spheredyn_v9.py",
    "tests/test_spheredyn_v9.py",
]
REPLACED = [
    "src/baselines/models.py",
    "scripts/spheredyn/evaluate_goal_reference_ensemble.py",
]
EDITS = [
    ("scripts/run_ttc.py",
     "from src.ttc.memory import EligibleMemory, assert_no_future_targets",
     "from src.spherettc import EligibleMemory, assert_no_future_targets", 1),
    ("scripts/run_ttc.py",
     "from src.ttc.sphere import SphereTTCConfig, SphereTTCCalibrator",
     "from src.spherettc import SphereTTCConfig, SphereTTCCalibrator", 1),
    ("scripts/run_ttc.py",
     "def _eligible_from_buffer(buffer, current, memory_size):\n"
     "    current = np.datetime64(current)\n"
     "    eligible = [item for item in buffer if np.datetime64(item[0]) <= current]\n"
     "    if memory_size is not None:\n"
     "        eligible = eligible[-int(memory_size) :]\n"
     "    return eligible\n",
     "# The causal memory-admission rule lives in src/spherettc.py (single source of truth).\n"
     "from src.spherettc import eligible_from_buffer as _eligible_from_buffer  # noqa: E402\n", 1),
    ("src/ttc/legacy/sphere_v2.py", "from src.ttc.sphere import _complex_delta_clamp",
     "from src.spherettc import _complex_delta_clamp", 1),
    ("src/ttc/legacy/sphere_v3.py", "from src.ttc.sphere import _complex_delta_clamp",
     "from src.spherettc import _complex_delta_clamp", 1),
    ("src/ttc/legacy/sphere_v4.py", "from src.ttc.sphere import _complex_delta_clamp",
     "from src.spherettc import _complex_delta_clamp", 1),
    ("tests/test_sphere_ttc.py", "from src.ttc.sphere import SphereTTCConfig, SphereTTCCalibrator",
     "from src.spherettc import SphereTTCConfig, SphereTTCCalibrator", 1),
    ("tests/test_no_leakage_memory.py", "from src.ttc.memory import EligibleMemory",
     "from src.spherettc import EligibleMemory", 1),
    ("src/ttc/__init__.py", '"""Test-time calibration methods."""',
     '"""Evaluation metrics, comparison TTC methods and legacy SphereTTC variants.\n\n'
     'The final SphereTTC module is src/spherettc.py.\n"""', 1),
]

MODELS_DOCSTRING = (
    '"""Baseline backbones trained under our protocol and the model factory.\n\n'
    "ConvLSTM, FNO, grid Transformer and ViT live here; CirT and the ClimODE-style\n"
    "adapter live in external_models.py.  The proposed SphereDyn model lives in\n"
    "src/spheredyn.py (the pre-2026-10-03 version of this file, which also held the\n"
    "SphereDyn building blocks, is archived in archive/pre_consolidation_20261003/).\n"
    '"""\n\n'
)
EVAL_IMPORT = (
    "\n# The SphereTTC-v22 (Mode B) math lives in src/spherettc.py; it is re-exported\n"
    "# here for scripts/spheredyn/run_main_spherettc_v2.py.\n"
    "from src.spherettc import (  # noqa: F401\n"
    "    REFERENCES,\n"
    "    RIDGE_RATIOS,\n"
    "    SPATIAL_SHRINKAGES,\n"
    "    _candidate_weights,\n"
    "    _energy,\n"
    "    _feature_statistics,\n"
    "    _fit_backbone,\n"
    "    _ridge_weights,\n"
    ")\n"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_top_level(text: str, names: set[str]) -> str:
    """Remove top-level defs/assignments by name, together with the blank lines before them."""
    lines = text.splitlines(keepends=True)
    drop: set[int] = set()
    for node in ast.parse(text).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            start = node.lineno - 1
            while start > 0 and lines[start - 1].strip() == "":
                start -= 1
            drop.update(range(start, node.end_lineno))
    return "".join(line for index, line in enumerate(lines) if index not in drop)


def regenerate_models(original: str) -> str:
    text = _remove_top_level(
        original,
        {"_spherical_pad", "SphericalDepthwiseBlock", "StableSphericalBandDynamics", "SphereDynForecast"},
    )
    replacements = [
        ("from torch_harmonics import InverseRealSHT, RealSHT\n", ""),
        ("        from .spheredyn_v9 import SphereDynV9Forecast\n\n        return SphereDynV9Forecast(\n",
         "        from ..spheredyn import SphereDyn\n\n        return SphereDyn(\n"),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise SystemExit(f"models.py: expected one {old[:40]!r}")
        text = text.replace(old, new)
    return MODELS_DOCSTRING + text


def regenerate_evaluate(original: str) -> str:
    text = _remove_top_level(
        original,
        {"REFERENCES", "RIDGE_RATIOS", "SPATIAL_SHRINKAGES", "_ridge_weights", "_energy",
         "_candidate_weights", "_feature_statistics", "_fit_backbone"},
    )
    anchor = "from src.experiments.protocol import COMMON_EVALUATION_VARIABLES\n"
    if text.count(anchor) != 1:
        raise SystemExit("evaluate_goal_reference_ensemble.py: import anchor not found")
    return text.replace(anchor, anchor + EVAL_IMPORT, 1)


def main() -> None:
    if LOG.exists():
        raise SystemExit(f"already consolidated: {LOG}")
    for relative in REQUIRED_NEW:
        if not (ROOT / relative).is_file():
            raise SystemExit(f"missing new canonical file: {relative}")
    for relative in MOVED + REPLACED:
        if not (ROOT / relative).is_file():
            raise SystemExit(f"missing original: {relative}")
        if (ROOT / ARCHIVE / relative).exists():
            raise SystemExit(f"archive destination exists: {ARCHIVE}/{relative}")
    for relative, old, _, count in EDITS:
        found = (ROOT / relative).read_text(encoding="utf-8").count(old)
        if found != count:
            raise SystemExit(f"{relative}: expected {count} x {old[:50]!r}, found {found}")

    moves, replaced, edits = [], [], []
    originals = {r: (ROOT / r).read_text(encoding="utf-8") for r in REPLACED}
    for relative in MOVED + REPLACED:
        source, target = ROOT / relative, ROOT / ARCHIVE / relative
        digest = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        if sha(target) != digest:
            raise SystemExit(f"content changed while archiving {relative}")
        moves.append({"from": relative, "to": f"{ARCHIVE}/{relative}", "sha256": digest})

    generated = {
        "src/baselines/models.py": regenerate_models(originals["src/baselines/models.py"]),
        "scripts/spheredyn/evaluate_goal_reference_ensemble.py": regenerate_evaluate(
            originals["scripts/spheredyn/evaluate_goal_reference_ensemble.py"]
        ),
    }
    for relative, text in generated.items():
        path = ROOT / relative
        if path.exists():
            raise SystemExit(f"replacement target exists: {relative}")
        path.write_text(text, encoding="utf-8")
        replaced.append({"file": relative, "original_archived_as": f"{ARCHIVE}/{relative}", "sha256": sha(path)})

    for relative, old, new, count in EDITS:
        path = ROOT / relative
        before = sha(path)
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
        edits.append({"file": relative, "old": old, "new": new, "count": count,
                      "sha256_before": before, "sha256_after": sha(path)})

    LOG.write_text(json.dumps({
        "consolidated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "SphereTTC consolidated into src/spherettc.py and SphereDyn into "
                       "src/spheredyn.py; superseded originals archived byte-identically.",
        "moves": moves,
        "edits": edits,
        "replaced": replaced,
        "added_files": REQUIRED_NEW,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"CONSOLIDATION_OK moves={len(moves)} replaced={len(replaced)} edits={len(edits)}")


if __name__ == "__main__":
    main()
