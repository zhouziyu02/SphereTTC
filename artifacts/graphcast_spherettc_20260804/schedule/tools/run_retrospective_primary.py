#!/usr/bin/env python
"""Post-freeze retrospective check of the frozen global primary on 2018--2019."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import search_schedule as search


ROOT = Path(__file__).resolve().parents[1]


def find_repository() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts/run_ttc.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the migrated SOON repository root")


SOURCE = find_repository()
CACHE_ROOT = SOURCE / "artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast"
TRUTH = SOURCE / "data/S2S"
CLIMATOLOGY = SOURCE / "artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz"


def run(task: str, command: list[str]) -> None:
    log_path = ROOT / "logs" / f"{task}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w") as log:
        log.write("started_at=" + datetime.now(timezone.utc).isoformat() + "\n")
        log.write("physical_gpu=1\ncommand=" + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write("finished_at=" + datetime.now(timezone.utc).isoformat() + "\n")
        log.write(f"returncode={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"{task} failed; see {log_path}")


def base(year: int, metrics: Path, profile: Path) -> list[str]:
    return [
        sys.executable,
        str(SOURCE / "scripts/run_ttc.py"),
        "--config", str(SOURCE / "configs/unified_11model_ttc.yaml"),
        "--model", "graphcast",
        "--cache", str(CACHE_ROOT / f"forecast_cache_{year}_121x240.zarr"),
        "--truth-root", str(TRUTH),
        "--truth-workers", "4",
        "--acc-climatology", str(CLIMATOLOGY),
        "--no-cache-output",
        "--metrics-output", str(metrics),
        "--profile-output", str(profile),
    ]


def concatenate(paths: list[Path]) -> dict[str, np.ndarray]:
    items = [search.load(path) for path in paths]
    result = {"lead_time": items[0]["lead_time"], "variable": items[0]["variable"]}
    for name in items[0]:
        if name in result:
            continue
        result[name] = np.concatenate([item[name] for item in items], axis=0)
    return result


def main() -> None:
    primary = json.loads((ROOT / "FROZEN_PRIMARY_GLOBAL_2017.json").read_text())
    if primary["status"] != "FROZEN_BEFORE_2020_METRICS":
        raise RuntimeError("primary is not frozen")
    output = ROOT / "metrics/retrospective_2018_2019"
    output.mkdir(parents=True, exist_ok=True)
    for year in (2018, 2019):
        raw_path = output / f"raw_{year}.npz"
        ttc_path = output / f"{primary['selected_candidate_id']}_{year}.npz"
        run(
            f"retrospective_raw_{year}",
            base(year, raw_path, ROOT / f"profiles/retrospective_raw_{year}.json")
            + ["--method", "raw"],
        )
        run(
            f"retrospective_{primary['selected_candidate_id']}_{year}",
            base(year, ttc_path, ROOT / f"profiles/retrospective_{primary['selected_candidate_id']}_{year}.json")
            + [
                "--method", "sphere_ttc",
                "--params-json", str(ROOT / f"configs/generated/{primary['selected_candidate_id']}.json"),
                "--warmup-cache", str(CACHE_ROOT / f"forecast_cache_{year - 1}_121x240.zarr"),
                "--trim-warmup-cache",
                "--warmup-truth-root", str(TRUTH),
            ],
        )
    raw = concatenate([output / "raw_2018.npz", output / "raw_2019.npz"])
    calibrated = concatenate(
        [
            output / f"{primary['selected_candidate_id']}_2018.npz",
            output / f"{primary['selected_candidate_id']}_2019.npz",
        ]
    )
    summary = search.summarize(raw, calibrated, 0, len(raw["init_time"]))
    payload = {
        "role": "post-freeze_secondary_check_not_used_for_selection",
        "primary_frozen_at": primary["created_at"],
        "selected_candidate_id": primary["selected_candidate_id"],
        **summary,
    }
    (ROOT / "RETROSPECTIVE_2018_2019_PRIMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "RETROSPECTIVE_PRIMARY_COMPLETE",
        f"rmse_gain={100 * summary['mean_relative_rmse_gain']:.4f}%",
        f"acc_delta={summary['mean_acc_delta']:+.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
