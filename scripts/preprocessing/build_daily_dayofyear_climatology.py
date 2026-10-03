#!/usr/bin/env python
"""Build a leakage-free daily day-of-year climatology from the fit split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.data_sources.s2s_daily import PROTOCOL_VARIABLES, load_coordinates, load_day
from src.utils.climatology import calendar_day_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-root", required=True)
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--fit-start", default="1979-01-01")
    parser.add_argument("--fit-end", default="2016-12-31")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    missing = [name for name in args.variables if name not in PROTOCOL_VARIABLES]
    if missing:
        raise RuntimeError(f"unknown variables: {missing}")
    variable_indices = [PROTOCOL_VARIABLES.index(name) for name in args.variables]
    dates = np.arange(
        np.datetime64(args.fit_start, "D"),
        np.datetime64(args.fit_end, "D") + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )
    day_indices = calendar_day_index(dates)
    lat, lon = load_coordinates(args.truth_root)
    sums = np.zeros((366, len(args.variables), len(lat), len(lon)), dtype=np.float64)
    counts = np.zeros(366, dtype=np.int32)

    workers = max(1, int(args.workers))
    batch_size = max(1, int(args.batch_size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(dates), batch_size):
            stop = min(start + batch_size, len(dates))
            values = list(
                pool.map(
                    lambda day: load_day(args.truth_root, day)[variable_indices],
                    dates[start:stop],
                )
            )
            for value, day_index in zip(values, day_indices[start:stop]):
                sums[day_index] += value.astype(np.float64)
                counts[day_index] += 1
            if stop % 512 == 0 or stop == len(dates):
                print(f"climatology loaded {stop}/{len(dates)} days", flush=True)

    # Only leap years contribute directly to February 29. This is intentional;
    # every other calendar day has one sample per fit year.
    if np.any(counts == 0):
        raise RuntimeError(f"empty calendar days: {np.where(counts == 0)[0].tolist()}")
    climatology = (sums / counts[:, None, None, None]).astype(np.float32)
    if not np.isfinite(climatology).all():
        raise RuntimeError("non-finite daily climatology")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        climatology=climatology,
        calendar_day=np.arange(1, 367, dtype=np.int16),
        variable=np.asarray(args.variables),
        lat=np.asarray(lat, dtype=np.float64),
        lon=np.asarray(lon, dtype=np.float64),
        sample_count=counts,
        fit_start=np.asarray(args.fit_start),
        fit_end=np.asarray(args.fit_end),
        definition=np.asarray(
            "daily month/day climatological mean on a 366-day leap calendar; "
            "fit split only; February 29 uses leap years only"
        ),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output.resolve()),
        "truth_root": str(Path(args.truth_root).resolve()),
        "fit_start": args.fit_start,
        "fit_end": args.fit_end,
        "days_loaded": int(len(dates)),
        "variables": args.variables,
        "calendar_day_sample_count": counts.tolist(),
        "definition": (
            "Daily month/day climatological mean on a 366-day leap calendar, "
            "computed only from the 1979-2016 fit split."
        ),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
