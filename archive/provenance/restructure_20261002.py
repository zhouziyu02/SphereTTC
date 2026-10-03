#!/usr/bin/env python3
"""One-shot in-place restructure of the SOON repository (2026-10-02).

Every move is non-overwriting and every text edit is an exact, counted string
replacement.  The full log (moves, edits, before/after SHA-256) is written to
RESTRUCTURE_LOG_20261002.json so that verify_migration.py can reverse the edits
and prove byte-identity against the 2026-08-06 migration manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "archive/provenance/RESTRUCTURE_LOG_20261002.json"

MOVES = [
    # Abandoned SphereTTC iterations (still importable for run_ttc.py --method sphere_ttc_v2..v7)
    *[(f"src/ttc/sphere_v{i}.py", f"src/ttc/legacy/sphere_v{i}.py") for i in range(2, 8)],
    # Alternative TTC methods kept as comparison baselines
    ("src/ttc/geo.py", "src/ttc/comparators/geo.py"),
    ("src/ttc/spectral.py", "src/ttc/comparators/spectral.py"),
    ("src/ttc/st_ttc.py", "src/ttc/comparators/st_ttc.py"),
    # SphereDyn + SphereTTC-v22 pipeline
    ("scripts/infer_spatiotemporal_checkpoint.py", "scripts/spheredyn/infer_spatiotemporal_checkpoint.py"),
    ("scripts/evaluate_goal_reference_ensemble.py", "scripts/spheredyn/evaluate_goal_reference_ensemble.py"),
    ("scripts/run_main_spherettc_v2.py", "scripts/spheredyn/run_main_spherettc_v2.py"),
    ("scripts/assess_main_prediction_goal.py", "scripts/spheredyn/assess_main_prediction_goal.py"),
    ("scripts/merge_metric_npz.py", "scripts/spheredyn/merge_metric_npz.py"),
    ("scripts/build_spheredyn_main_rmse_acc_table.py", "scripts/spheredyn/build_spheredyn_main_rmse_acc_table.py"),
    ("scripts/run_main_prediction_v2_4gpu.sh", "scripts/spheredyn/run_main_prediction_v2_4gpu.sh"),
    ("scripts/compute_spheredyn_main_rmse_acc_4gpu.sh", "scripts/spheredyn/compute_spheredyn_main_rmse_acc_4gpu.sh"),
    # Table generation
    ("scripts/render_main_results_49_variables_zh.py", "scripts/tables/render_main_results_49_variables_zh.py"),
    ("scripts/build_latest_results.py", "scripts/tables/build_latest_results.py"),
    ("scripts/summarize_publication_main_table.py", "scripts/tables/summarize_publication_main_table.py"),
    # Data preparation
    ("scripts/build_daily_dayofyear_climatology.py", "scripts/data/build_daily_dayofyear_climatology.py"),
    # Verifiers that only work inside the original ~14 GB server archive (kept byte-identical)
    ("scripts/verify_main_results_reproducibility.py", "archive/full_archive_verifiers/verify_main_results_reproducibility.py"),
    ("scripts/verify_latest_results_reproducibility.py", "archive/full_archive_verifiers/verify_latest_results_reproducibility.py"),
    ("scripts/build_latest_reproducibility_manifest.py", "archive/full_archive_verifiers/build_latest_reproducibility_manifest.py"),
    # Superseded top-level documents / environment files
    ("README.md", "archive/old_docs/README_full_archive_20260804.md"),
    ("README_ZH.md", "archive/old_docs/README_ZH_lightweight_migration_20260806.md"),
    ("PROJECT_PROPOSAL.md", "docs/PROJECT_PROPOSAL.md"),
    ("requirements.txt", "archive/old_env/requirements_original.txt"),
    ("sitecustomize.py", "archive/old_env/sitecustomize.py"),
    ("requirements-repro.txt", "env/requirements-verify.txt"),
    ("requirements-gpu.txt", "env/requirements-gpu-graphcast.txt"),
    ("requirements_external_baselines.txt", "env/requirements-external-baselines.txt"),
    ("MIGRATION_MANIFEST.sha256", "archive/provenance/MIGRATION_MANIFEST_20260806.sha256"),
    ("verify_migration.py", "archive/provenance/verify_migration_20260806.py"),
]

LEGACY = "from src.ttc.legacy.sphere_v"
SPHERE_V = "from src.ttc.sphere_v"
EDITS = [
    # (path after move, old, new, expected count)
    ("scripts/run_ttc.py", SPHERE_V, LEGACY, 6),
    ("scripts/run_ttc.py", "from src.ttc.geo import", "from src.ttc.comparators.geo import", 1),
    ("scripts/run_ttc.py", "from src.ttc.st_ttc import", "from src.ttc.comparators.st_ttc import", 1),
    ("scripts/run_ttc.py", "from src.ttc.spectral import", "from src.ttc.comparators.spectral import", 1),
    ("src/ttc/legacy/sphere_v5.py", SPHERE_V, LEGACY, 1),
    ("src/ttc/legacy/sphere_v6.py", SPHERE_V, LEGACY, 1),
    ("src/ttc/legacy/sphere_v7.py", SPHERE_V, LEGACY, 2),
    ("scripts/spheredyn/infer_spatiotemporal_checkpoint.py", "parents[1]", "parents[2]", 1),
    ("scripts/spheredyn/evaluate_goal_reference_ensemble.py", "parents[1]", "parents[2]", 1),
    ("scripts/spheredyn/run_main_spherettc_v2.py", "parents[1]", "parents[2]", 1),
    ("scripts/spheredyn/run_main_spherettc_v2.py", "from scripts.evaluate_goal_reference_ensemble import", "from scripts.spheredyn.evaluate_goal_reference_ensemble import", 1),
    ("scripts/spheredyn/run_main_spherettc_v2.py", "from scripts.infer_spatiotemporal_checkpoint import", "from scripts.spheredyn.infer_spatiotemporal_checkpoint import", 1),
    ("scripts/spheredyn/build_spheredyn_main_rmse_acc_table.py", "parents[1]", "parents[2]", 1),
    ("scripts/spheredyn/run_main_prediction_v2_4gpu.sh", 'ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)', 'ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)', 1),
    ("scripts/spheredyn/run_main_prediction_v2_4gpu.sh", "scripts/infer_spatiotemporal_checkpoint.py", "scripts/spheredyn/infer_spatiotemporal_checkpoint.py", 1),
    ("scripts/spheredyn/run_main_prediction_v2_4gpu.sh", "scripts/run_main_spherettc_v2.py", "scripts/spheredyn/run_main_spherettc_v2.py", 1),
    ("scripts/spheredyn/run_main_prediction_v2_4gpu.sh", "scripts/assess_main_prediction_goal.py", "scripts/spheredyn/assess_main_prediction_goal.py", 1),
    ("scripts/spheredyn/compute_spheredyn_main_rmse_acc_4gpu.sh", 'ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)', 'ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)', 1),
    ("scripts/spheredyn/compute_spheredyn_main_rmse_acc_4gpu.sh", "scripts/merge_metric_npz.py", "scripts/spheredyn/merge_metric_npz.py", 2),
    ("scripts/spheredyn/compute_spheredyn_main_rmse_acc_4gpu.sh", "scripts/build_spheredyn_main_rmse_acc_table.py", "scripts/spheredyn/build_spheredyn_main_rmse_acc_table.py", 1),
    ("scripts/tables/render_main_results_49_variables_zh.py", "parents[1]", "parents[2]", 1),
    ("scripts/tables/build_latest_results.py", "parents[1]", "parents[2]", 1),
    ("scripts/tables/build_latest_results.py", "from scripts.render_main_results_49_variables_zh import", "from scripts.tables.render_main_results_49_variables_zh import", 1),
    ("scripts/data/build_daily_dayofyear_climatology.py", "parents[1]", "parents[2]", 1),
    ("tests/test_main_prediction_goal_v2.py", "scripts/assess_main_prediction_goal.py", "scripts/spheredyn/assess_main_prediction_goal.py", 1),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if LOG.exists():
        raise SystemExit(f"already restructured: {LOG}")
    # Preconditions: every source exists, no destination exists.
    for src, dst in MOVES:
        if not (ROOT / src).is_file():
            raise SystemExit(f"missing source: {src}")
        if (ROOT / dst).exists():
            raise SystemExit(f"destination exists: {dst}")
    moves = []
    for src, dst in MOVES:
        s, d = ROOT / src, ROOT / dst
        digest = sha(s)
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():  # never overwrite
            raise SystemExit(f"destination appeared: {dst}")
        os.rename(s, d)
        if sha(d) != digest:
            raise SystemExit(f"content changed during move: {dst}")
        moves.append({"from": src, "to": dst, "sha256": digest})
    edits = []
    for rel, old, new, count in EDITS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        found = text.count(old)
        if found != count:
            raise SystemExit(f"{rel}: expected {count} x {old!r}, found {found}")
        before = sha(path)
        path.write_text(text.replace(old, new), encoding="utf-8")
        edits.append({"file": rel, "old": old, "new": new, "count": count,
                      "sha256_before": before, "sha256_after": sha(path)})
    LOG.write_text(json.dumps({
        "restructured_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "In-place restructure: legacy SphereTTC iterations -> src/ttc/legacy, "
                       "TTC comparators -> src/ttc/comparators, scripts grouped by pipeline, "
                       "superseded docs/env/full-archive verifiers -> archive/. Frozen artifacts/ untouched.",
        "snapshot": "archive/ORIGINAL_SNAPSHOT_20261002.tar.gz",
        "moves": moves,
        "edits": edits,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RESTRUCTURE_OK moves={len(moves)} edits={len(edits)}")


if __name__ == "__main__":
    main()
