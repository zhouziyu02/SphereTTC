#!/usr/bin/env python
"""Hash immutable SOON code/assets before and after the isolated experiment."""

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
EXTRA = (
    "README.md",
    "PROJECT_PROPOSAL.md",
    "external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz",
    "external/official_checkpoints/graphcast/stats/mean_by_level.nc",
    "external/official_checkpoints/graphcast/stats/diffs_stddev_by_level.nc",
    "external/official_checkpoints/graphcast/stats/stddev_by_level.nc",
    "artifacts/spheredyn_spherettc_open_goal_20260801/MAIN_REPRODUCIBILITY_MANIFEST.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot() -> dict[str, object]:
    paths: list[Path] = []
    for relative in ("src", "scripts", "configs"):
        paths.extend(path for path in (SOURCE / relative).rglob("*") if path.is_file())
    paths.extend(SOURCE / relative for relative in EXTRA)
    hashes = {
        str(path.relative_to(SOURCE)): sha256(path)
        for path in sorted(set(paths))
    }
    unexpected = [
        name
        for name in ("cache", "profiles")
        if (SOURCE / name).exists()
    ]
    return {
        "source_root": str(SOURCE),
        "sha256": hashes,
        "unexpected_experiment_directories_in_source": unexpected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    current = snapshot()
    path = OUT / f"source_{args.phase}.json"
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    if current["unexpected_experiment_directories_in_source"]:
        raise SystemExit("UNEXPECTED_EXPERIMENT_DIRECTORY_IN_SOURCE")
    if args.phase == "after":
        before = json.loads((OUT / "source_before.json").read_text())
        unchanged = current == before
        (OUT / "comparison.json").write_text(
            json.dumps({"source_repository_unchanged": unchanged}, indent=2) + "\n"
        )
        if not unchanged:
            raise SystemExit("SOURCE_REPOSITORY_CHANGED")
        print("SOURCE_REPOSITORY_UNCHANGED")
    else:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
