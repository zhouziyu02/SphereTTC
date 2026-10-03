#!/usr/bin/env python
"""Create compact file and tree hashes for the consolidated latest experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT = REPOSITORY / "artifacts/graphcast_spherettc_20260804"
OUTPUT = ARTIFACT / "LATEST_REPRODUCIBILITY_MANIFEST.json"
FILES = (
    "README.md",
    "pytest.ini",
    "requirements-repro.txt",
    "requirements-gpu.txt",
    "scripts/build_latest_results.py",
    "scripts/build_latest_reproducibility_manifest.py",
    "scripts/verify_latest_results_reproducibility.py",
    "scripts/verify_main_results_reproducibility.py",
    "artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv",
    "artifacts/graphcast_spherettc_20260804/GRAPHCAST_L64_S08_2018_2019.csv",
    "artifacts/graphcast_spherettc_20260804/LATEST_RESULTS_49_VARIABLES_ZH.md",
    "artifacts/graphcast_spherettc_20260804/COMPARISON_WITH_FROZEN_49_TABLES.json",
    "artifacts/graphcast_spherettc_20260804/LATEST_EXPERIMENT_SUMMARY_ZH.md",
    "artifacts/graphcast_spherettc_20260804/README.md",
    "artifacts/graphcast_spherettc_20260804/REPRODUCIBILITY.md",
    "artifacts/graphcast_spherettc_20260804/ENVIRONMENT_20260804.txt",
    "external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz",
    "external/official_checkpoints/graphcast/stats/mean_by_level.nc",
    "external/official_checkpoints/graphcast/stats/diffs_stddev_by_level.nc",
    "external/official_checkpoints/graphcast/stats/stddev_by_level.nc",
)
TREES = (
    "src",
    "scripts",
    "configs",
    "external/graphcast_official",
    "external/official_checkpoints/graphcast",
    "artifacts/graphcast_spherettc_20260804/parameter_only",
    "artifacts/graphcast_spherettc_20260804/schedule",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_record(path: Path) -> dict[str, int | str]:
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
    return {
        "sha256": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total,
    }


def main() -> None:
    payload = {
        "status": "FROZEN_LATEST_RESULTS_REPRODUCIBLE",
        "scope": "original 49 tables plus GraphCast+SphereTTC experiments frozen through 2026-08-04",
        "quick_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/verify_latest_results_reproducibility.py",
        "full_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/verify_latest_results_reproducibility.py --full-integrity",
        "sha256": {relative: sha256(REPOSITORY / relative) for relative in FILES},
        "tree_sha256": {
            relative: tree_record(REPOSITORY / relative) for relative in TREES
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "LATEST_MANIFEST_BUILT",
        *(f"{name}={record['file_count']}files/{record['total_bytes']}bytes" for name, record in payload["tree_sha256"].items()),
    )


if __name__ == "__main__":
    main()
