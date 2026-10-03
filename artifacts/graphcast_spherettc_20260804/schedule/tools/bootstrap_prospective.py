#!/usr/bin/env python
"""Circular moving-block bootstrap for the frozen 2020 prospective results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPLICATES = 2000
BLOCK_DAYS = 14
SEED = 20260804


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def circular_sums(values: np.ndarray, length: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for offset in range(length):
        result += np.roll(values, -offset, axis=0)
    return result


def bootstrap_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    count = values.shape[0]
    full_count, remainder = divmod(count, BLOCK_DAYS)
    full = circular_sums(values, BLOCK_DAYS)
    result = full[starts[:, :full_count]].sum(axis=1)
    if remainder:
        tail = circular_sums(values, remainder)
        result += tail[starts[:, full_count]]
    return result


def evaluate(
    raw: dict[str, np.ndarray], calibrated: dict[str, np.ndarray], starts: np.ndarray
) -> dict[str, object]:
    count = len(raw["init_time"])
    raw_mse = bootstrap_sum(raw["mse"], starts) / count
    calibrated_mse = bootstrap_sum(calibrated["mse"], starts) / count
    cell_gain = 1.0 - np.sqrt(calibrated_mse) / np.sqrt(raw_mse)
    mean_gain = cell_gain.mean(axis=(1, 2))
    negative_fraction = (cell_gain < 0).mean(axis=(1, 2))

    def acc(data: dict[str, np.ndarray]) -> np.ndarray:
        numerator = bootstrap_sum(data["acc_numerator"], starts)
        prediction = bootstrap_sum(data["acc_prediction_energy"], starts)
        target = bootstrap_sum(data["acc_target_energy"], starts)
        return numerator / np.sqrt(np.maximum(prediction * target, 1e-24))

    mean_acc_delta = (acc(calibrated) - acc(raw)).mean(axis=(1, 2))
    passed = (mean_gain >= 0.02) & (negative_fraction <= 0.05) & (mean_acc_delta >= 0.0)

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 50.0, 97.5])]

    return {
        "mean_relative_rmse_gain_percentiles_2p5_50_97p5": interval(mean_gain),
        "mean_acc_delta_percentiles_2p5_50_97p5": interval(mean_acc_delta),
        "negative_rmse_cell_fraction_percentiles_2p5_50_97p5": interval(negative_fraction),
        "bootstrap_success_probability": float(passed.mean()),
    }


def main() -> None:
    metrics = ROOT / "metrics/holdout_2020"
    raw = load(metrics / "raw.npz")
    primary = load(metrics / "primary_global_sphere_ttc.npz")
    secondary = load(metrics / "scheduled_sphere_ttc.npz")
    count = len(raw["init_time"])
    block_count = (count + BLOCK_DAYS - 1) // BLOCK_DAYS
    starts = np.random.default_rng(SEED).integers(
        0, count, size=(REPLICATES, block_count), endpoint=False
    )
    payload = {
        "method": "circular moving-block bootstrap over initialization dates",
        "replicates": REPLICATES,
        "block_days": BLOCK_DAYS,
        "seed": SEED,
        "primary_global": evaluate(raw, primary, starts),
        "secondary_family_lead_schedule": evaluate(raw, secondary, starts),
    }
    (ROOT / "PROSPECTIVE_2020_BOOTSTRAP.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("PROSPECTIVE_BOOTSTRAP_COMPLETE", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
