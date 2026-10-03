#!/usr/bin/env python
"""Run the unmodified official GraphCast adapter into the isolated directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def find_repository() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts/run_ttc.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the migrated SOON repository root")


SOURCE = find_repository()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--init-start", default=None)
    parser.add_argument("--max-init-times", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    status_path = args.status.resolve()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    import src.data_sources.graphcast_gencast_official as graphcast_adapter
    from src.utils.config import load_config

    cfg = load_config("configs/official_daily_common49_2018_2019.yaml")
    cfg["experiment"]["year"] = int(args.year)
    if args.max_init_times is not None:
        cfg["experiment"]["max_init_times"] = int(args.max_init_times)
    if args.init_start is not None:
        original_init_times = graphcast_adapter._init_times

        def restricted_init_times(config):
            values = original_init_times(config)
            return values[values >= np.datetime64(args.init_start, "ns")]

        graphcast_adapter._init_times = restricted_init_times
    status = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "year": int(args.year),
        "max_init_times": args.max_init_times,
        "output": str(output_path),
        "init_start_requested": args.init_start,
        "official_asset_status": graphcast_adapter.inspect_graphcast_official(),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    dataset = graphcast_adapter.build_graphcast_cache(cfg, str(output_path))
    status.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "shape": list(dataset["pred"].shape),
            "init_start": str(dataset.init_time.values[0]),
            "init_end": str(dataset.init_time.values[-1]),
            "complete": True,
        }
    )
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print("GRAPHCAST_HOLDOUT_COMPLETE", status["shape"], flush=True)


if __name__ == "__main__":
    main()
