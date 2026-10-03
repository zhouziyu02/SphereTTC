#!/usr/bin/env python
"""Verify and regenerate the frozen 49-variable main-result document.

This performs aggregation only. It never trains a model or runs forecast inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.render_main_results_49_variables_zh import _load_rows, _render


MODELS = (
    "convlstm",
    "transformer",
    "fno",
    "vit",
    "cirt",
    "climode",
    "fourcastnetv2",
    "oneforecast",
    "fuxi",
    "pangu",
    "graphcast",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    repository = REPOSITORY
    default_root = repository / (
        "artifacts/spheredyn_spherettc_open_goal_20260801"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "MAIN_REPRODUCIBILITY_MANIFEST.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path at which to write the byte-identical regenerated Markdown.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen file is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )

    baseline_source = repository / manifest["table_sources"]["baseline_csv"]
    spheredyn_source = repository / manifest["table_sources"]["spheredyn_csv"]
    canonical_document = repository / manifest["canonical_document"]
    baseline_rows = _read_csv(baseline_source)
    spheredyn_rows = _read_csv(spheredyn_source)
    if len(baseline_rows) != 5390 or len(spheredyn_rows) != 490:
        raise RuntimeError(
            "unexpected frozen row counts: "
            f"baseline={len(baseline_rows)}, SphereDyn={len(spheredyn_rows)}"
        )

    with tempfile.TemporaryDirectory(prefix="soon-main-repro-") as temporary:
        temporary_root = Path(temporary)
        baseline_output = temporary_root / "baseline"
        subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/summarize_publication_main_table.py"),
                "--metrics-root",
                str(
                    repository
                    / "artifacts/ttc_publication_corrected_20260726/metrics"
                ),
                "--models",
                *MODELS,
                "--output-root",
                str(baseline_output),
            ],
            cwd=repository,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        generated_baseline = baseline_output / "MAIN_RESULTS.csv"
        generated_spheredyn = temporary_root / "SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv"
        subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/build_spheredyn_main_rmse_acc_table.py"),
                "--raw",
                str(
                    repository
                    / (
                        "artifacts/spheredyn_spherettc_open_goal_20260801/"
                        "main_prediction_v2_seed44/paper_metrics/"
                        "spheredyn_v9_raw_2018_2019.npz"
                    )
                ),
                "--final",
                str(
                    repository
                    / (
                        "artifacts/spheredyn_spherettc_open_goal_20260801/"
                        "main_prediction_v2_seed44/paper_metrics/"
                        "spheredyn_v9_spherettc_v22_2018_2019.npz"
                    )
                ),
                "--output",
                str(generated_spheredyn),
            ],
            cwd=repository,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if generated_baseline.read_bytes() != baseline_source.read_bytes():
            raise RuntimeError("baseline CSV is not byte-reproducible from frozen metrics")
        if generated_spheredyn.read_bytes() != spheredyn_source.read_bytes():
            raise RuntimeError("SphereDyn CSV is not byte-reproducible from frozen metrics")

        variables, units, values = _load_rows(
            [generated_baseline, generated_spheredyn]
        )
        source_labels = [
            baseline_source.relative_to(repository),
            spheredyn_source.relative_to(repository),
        ]
        regenerated = _render(source_labels, variables, units, values)
        if regenerated.encode("utf-8") != canonical_document.read_bytes():
            raise RuntimeError("canonical Markdown is not byte-reproducible")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(regenerated, encoding="utf-8")

    print(
        "MAIN_RESULTS_REPRODUCIBLE "
        "baseline_rows=5390 spheredyn_rows=490 combined_rows=5880 "
        "variables=49 rows_per_table=24 markdown_byte_identical=true"
    )


if __name__ == "__main__":
    main()
