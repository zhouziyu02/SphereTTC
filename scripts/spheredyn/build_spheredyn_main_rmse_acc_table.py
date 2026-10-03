#!/usr/bin/env python
"""Build physical RMSE/ACC rows for the frozen SphereDyn v2 main run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ACC_KEYS = (
    "acc_numerator",
    "acc_prediction_energy",
    "acc_target_energy",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = ("init_time", "lead_time", "variable", "mse", *ACC_KEYS)
        missing = [key for key in required if key not in source]
        if missing:
            raise RuntimeError(f"{path} is missing {missing}")
        return {key: source[key] for key in source.files}


def _unit(variable: str) -> str:
    if variable.startswith("z"):
        return "m^2 s^-2"
    if variable.startswith("q"):
        return "kg kg^-1"
    if variable.startswith("t"):
        return "K"
    if variable.startswith(("u", "v")):
        return "m s^-1"
    if variable == "mslp":
        return "Pa"
    raise ValueError(f"unknown variable unit: {variable}")


def _aggregate(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rmse = np.sqrt(np.asarray(data["mse"], dtype=np.float64).mean(axis=0))
    numerator = np.asarray(data["acc_numerator"], dtype=np.float64).sum(axis=0)
    prediction_energy = np.asarray(
        data["acc_prediction_energy"], dtype=np.float64
    ).sum(axis=0)
    target_energy = np.asarray(data["acc_target_energy"], dtype=np.float64).sum(
        axis=0
    )
    denominator = np.sqrt(
        np.maximum(prediction_energy * target_energy, 1e-24)
    )
    acc = numerator / denominator
    if not np.isfinite(rmse).all() or not np.isfinite(acc).all():
        raise RuntimeError("SphereDyn RMSE/ACC contains non-finite values")
    return rmse, acc


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    default_root = repository / (
        "artifacts/spheredyn_spherettc_open_goal_20260801/"
        "main_prediction_v2_seed44/paper_metrics"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=default_root / "spheredyn_v9_raw_2018_2019.npz",
    )
    parser.add_argument(
        "--final",
        type=Path,
        default=default_root / "spheredyn_v9_spherettc_v22_2018_2019.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv",
    )
    args = parser.parse_args()

    raw = _load(args.raw)
    final = _load(args.final)
    for coordinate in ("init_time", "lead_time", "variable"):
        if not np.array_equal(raw[coordinate], final[coordinate]):
            raise RuntimeError(f"raw/final coordinate mismatch: {coordinate}")
    if raw["mse"].shape != (730, 5, 49):
        raise RuntimeError(f"unexpected main metric shape: {raw['mse'].shape}")
    expected_leads = np.asarray([24, 72, 120, 168, 240], dtype=np.int32)
    if not np.array_equal(raw["lead_time"], expected_leads):
        raise RuntimeError("unexpected lead times")

    raw_rmse, raw_acc = _aggregate(raw)
    final_rmse, final_acc = _aggregate(final)
    rows = []
    for method, rmse, acc in (
        ("raw", raw_rmse, raw_acc),
        ("sphere_ttc_v22", final_rmse, final_acc),
    ):
        for lead_index, lead in enumerate(raw["lead_time"]):
            for variable_index, variable_value in enumerate(raw["variable"]):
                variable = str(variable_value)
                rows.append(
                    {
                        "model": "spheredyn_v9",
                        "model_display_name": "SphereDyn-v9 (seed 44)",
                        "method": method,
                        "period": "2018-2019",
                        "lead_time_hours": int(lead),
                        "variable": variable,
                        "unit": _unit(variable),
                        "rmse": float(rmse[lead_index, variable_index]),
                        "acc": float(acc[lead_index, variable_index]),
                        "n_initializations": 730,
                    }
                )
    if len(rows) != 490:
        raise RuntimeError(f"unexpected SphereDyn table row count: {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.output.resolve()} rows={len(rows)}")


if __name__ == "__main__":
    main()
