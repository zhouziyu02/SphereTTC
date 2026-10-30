#!/usr/bin/env python3
"""Download only the ERA5 fields and dates needed by an experiment."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First UTC target date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Last UTC target date, inclusive.")
    parser.add_argument("--output", required=True, type=Path, help="Output truth.zarr path.")
    parser.add_argument("--source", choices=("main_daily", "conservative_6h"), default="main_daily",
                        help="main_daily matches the main archive; conservative_6h is a separate analysis source.")
    parser.add_argument("--include-10hpa", action="store_true", help="Prepare 54 instead of 49 default fields.")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="Download only these short names (for example, z500 t850 t2m mslp u10). Overrides the default set.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing exact date prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Print scope without network access or writing data.")
    args = parser.parse_args()
    try:
        first, last = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if first > last:
            raise ValueError("start must not be after end")
    except ValueError as exc:
        parser.error(str(exc))
    variables = None
    if args.variables is not None:
        from spherettc.data import variable_source

        variables = tuple(args.variables)
        if len(set(variables)) != len(variables):
            parser.error("--variables must not contain duplicate field names")
        try:
            for variable in variables:
                variable_source(variable)
        except ValueError as exc:
            parser.error(str(exc))
    days = (last - first).days + 1
    count = len(variables) if variables is not None else (54 if args.include_10hpa else 49)
    print(f"Dates: {first} to {last} inclusive ({days} days)")
    print(f"Source: {args.source}; fields: {count}; grid: 121 x 240")
    print(f"Uncompressed output: {days * count * 121 * 240 * 4 / 2**30:.2f} GiB")
    print("Remote transfer depends on upstream chunk sizes and can greatly exceed output size.")
    print("Include forecast warmup and final verification dates in the requested interval.")
    if args.source == "main_daily":
        print("Uses published daily means, with exact 0.25-degree to 1.5-degree subsampling.")
    else:
        print("Requires all four UTC samples (00, 06, 12, 18) for each daily mean.")
    if args.dry_run:
        return
    from spherettc.data import canonical_variables, write_daily_store

    write_daily_store(args.output, args.start, args.end, args.source,
                      variables if variables is not None else canonical_variables(args.include_10hpa),
                      args.resume)


if __name__ == "__main__":
    main()
