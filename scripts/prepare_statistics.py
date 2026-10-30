#!/usr/bin/env python3
"""Build normalization and daily ACC climatology from training data only."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True, help="Downloaded truth.zarr store.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-start", default="1979-01-01")
    parser.add_argument("--fit-end", default="2016-12-31")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        first, last = date.fromisoformat(args.fit_start), date.fromisoformat(args.fit_end)
        if not date(1979, 1, 1) <= first <= last <= date(2016, 12, 31):
            raise ValueError("Statistics must use training dates within 1979--2016")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Fit interval: {first} to {last}, inclusive; no validation or test fallback")
    print(f"Output: {args.output_dir / 'stats.json'} and {args.output_dir / 'climatology.zarr'}")
    print("Population standard deviations use uniform grid weights; ACC climatology uses 366 month/day slots.")
    if args.dry_run:
        return
    from spherettc.data import prepare_statistics

    prepare_statistics(args.truth, args.output_dir, args.fit_start, args.fit_end)


if __name__ == "__main__":
    main()
