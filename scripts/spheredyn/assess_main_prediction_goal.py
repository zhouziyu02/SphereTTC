#!/usr/bin/env python
"""Assess the v2 single-seed main prediction goal without auxiliary analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _macro_nrmse(payload: dict[str, np.ndarray], std: np.ndarray) -> float:
    mse = np.asarray(payload["mse"], dtype=np.float64)
    if mse.ndim < 3:
        raise ValueError("mse must have [init_time, lead_time, variable] dimensions")
    value = np.sqrt(mse.mean(axis=0)) / std[None]
    if not np.all(np.isfinite(value)):
        raise ValueError("primary metric contains non-finite values")
    return float(value.mean())


def _validate_pair(
    raw: dict[str, np.ndarray], final: dict[str, np.ndarray]
) -> None:
    for coordinate in ("init_time", "lead_time", "variable"):
        if coordinate not in raw or coordinate not in final:
            raise ValueError(f"missing required coordinate {coordinate!r}")
        if not np.array_equal(raw[coordinate], final[coordinate]):
            raise ValueError(f"raw/final {coordinate} coordinates do not match")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--minimum-gain-percent", type=float, default=5.0)
    parser.add_argument("--graphcast", type=Path)
    args = parser.parse_args()

    raw = _load(args.raw)
    final = _load(args.final)
    _validate_pair(raw, final)
    stats = json.loads(args.stats.read_text())
    lookup = dict(zip(stats["variables"], stats["std"], strict=True))
    variables = [str(value) for value in raw["variable"]]
    missing = [name for name in variables if name not in lookup]
    if missing:
        raise ValueError(f"normalization statistics missing variables: {missing}")
    std = np.asarray([lookup[name] for name in variables], dtype=np.float64)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("normalization standard deviations must be finite and positive")

    raw_macro = _macro_nrmse(raw, std)
    final_macro = _macro_nrmse(final, std)
    gain = 100.0 * (1.0 - final_macro / raw_macro)
    passed = final_macro < raw_macro and gain >= args.minimum_gain_percent

    graphcast_metric = None
    if args.graphcast is not None:
        graphcast = _load(args.graphcast)
        _validate_pair(raw, graphcast)
        graphcast_metric = _macro_nrmse(graphcast, std)

    document = {
        "status": "MAIN_PREDICTION_GOAL_ACHIEVED" if passed else "MAIN_PREDICTION_GOAL_NOT_MET",
        "protocol": "artifacts/spheredyn_spherettc_open_goal_20260801/PROTOCOL_V2.json",
        "seed": args.seed,
        "raw_spheredyn_macro_nrmse": raw_macro,
        "final_spheredyn_plus_spherettc_macro_nrmse": final_macro,
        "gain_percent": gain,
        "minimum_gain_percent": args.minimum_gain_percent,
        "main_performance_gate_passed": passed,
        "graphcast_macro_nrmse": graphcast_metric,
        "graphcast_role": "reporting_only_reference",
        "graphcast_hard_gate": False,
        "multiseed_required": False,
        "ablation_required": False,
        "auxiliary_analysis_required": False,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
