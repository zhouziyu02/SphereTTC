#!/usr/bin/env python3
"""Run the archived main protocol on prepared forecast caches for all backbones."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from run_experiment import COMMON_VARIABLES, forecast_field

MODELS = (
    "convlstm", "transformer", "fno", "vit", "cirt", "climode",
    "fourcastnetv2", "oneforecast", "fuxi", "pangu", "graphcast",
)
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", type=Path, required=True,
                        help="Contains MODEL.zarr, with a 2017 warmup prefix and 2018--2019 forecasts.")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verification-delay-hours", type=float, default=0,
                        help="0: archived main protocol. 24: a separate completed-day replay.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without accessing data.")
    args = parser.parse_args()
    if len(args.models) != len(set(args.models)):
        parser.error("Each model must appear only once.")
    commands = []
    for model in args.models:
        commands.append([
            sys.executable, str(ROOT / "scripts/run_experiment.py"),
            "--model", model, "--params", str(ROOT / f"configs/params/{model}.json"),
            "--forecast-cache", str(args.forecast_dir / f"{model}.zarr"),
            "--truth", args.truth,
            "--climatology", str(args.statistics_dir / "climatology.zarr"),
            "--statistics", str(args.statistics_dir / "stats.json"),
            "--start", "2018-01-01", "--end", "2019-12-31", "--require-warmup",
            "--device", args.device,
            "--verification-delay-hours", str(args.verification_delay_hours),
            "--output", str(args.output / model),
        ])
    commands.append([
        sys.executable, str(ROOT / "scripts/summarize_results.py"), "--results",
        *[str(args.output / model / "results.json") for model in args.models],
        "--output", str(args.output / "summary"),
    ])
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new output directory.")
    missing = [args.forecast_dir / f"{model}.zarr" for model in args.models
               if not (args.forecast_dir / f"{model}.zarr").exists()]
    if missing:
        raise FileNotFoundError("Missing main forecast caches:\n" + "\n".join(map(str, missing)))
    # Check all forecast interfaces before starting any calibration run.
    for model in args.models:
        field, leads = forecast_field(str(args.forecast_dir / f"{model}.zarr"), COMMON_VARIABLES)
        if leads.tolist() != [24, 72, 120, 168, 240]:
            raise ValueError(f"{model}: main experiment requires leads 24,72,120,168,240 hours.")
        if (field.sizes["lat"], field.sizes["lon"]) != (121, 240):
            raise ValueError(f"{model}: main experiment requires the 121 x 240 grid.")
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
