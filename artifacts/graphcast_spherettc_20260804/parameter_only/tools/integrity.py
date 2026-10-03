#!/usr/bin/env python
"""Record and compare the read-only source repository's integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_repository() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts/run_ttc.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the migrated SOON repository root")


SOURCE = find_repository()
OUT = ROOT / "integrity"
HASH_ROOTS = ("src", "scripts", "configs")
HASH_FILES = (
    "README.md",
    "PROJECT_PROPOSAL.md",
    "artifacts/ttc_publication_corrected_20260726/params/graphcast.json",
    "artifacts/ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json",
    "artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz",
    "artifacts/ttc_publication_corrected_20260726/metrics/graphcast/raw/per_initialization_metrics.npz",
    "artifacts/ttc_publication_corrected_20260726/metrics/graphcast/sphere_ttc/per_initialization_metrics.npz",
    "artifacts/spheredyn_spherettc_open_goal_20260801/MAIN_REPRODUCIBILITY_MANIFEST.json",
)
CACHE_ROOT = "artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths: list[Path] = []
    for relative in HASH_ROOTS:
        paths.extend(path for path in (SOURCE / relative).rglob("*") if path.is_file())
    paths.extend(SOURCE / relative for relative in HASH_FILES)
    return {
        str(path.relative_to(SOURCE)): file_sha256(path)
        for path in sorted(set(paths))
    }


def cache_metadata_fingerprint() -> dict[str, object]:
    root = SOURCE / CACHE_ROOT
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        record = f"{path.relative_to(root)}\t{stat.st_size}\t{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
        count += 1
        total_bytes += stat.st_size
    return {
        "root": CACHE_ROOT,
        "file_count": count,
        "total_bytes": total_bytes,
        "path_size_mtime_sha256": digest.hexdigest(),
    }


def snapshot() -> dict[str, object]:
    return {
        "source_root": str(SOURCE),
        "hashed_files": source_hashes(),
        "graphcast_cache_metadata": cache_metadata_fingerprint(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    current = snapshot()
    path = OUT / f"source_{args.phase}.json"
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    if args.phase == "after":
        before = json.loads((OUT / "source_before.json").read_text())
        unchanged = current == before
        report = {
            "source_repository_unchanged": unchanged,
            "before": str(OUT / "source_before.json"),
            "after": str(path),
        }
        (OUT / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        if not unchanged:
            raise SystemExit("SOURCE_REPOSITORY_CHANGED")
        print("SOURCE_REPOSITORY_UNCHANGED")
    else:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
