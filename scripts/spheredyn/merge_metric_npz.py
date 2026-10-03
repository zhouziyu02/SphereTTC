#!/usr/bin/env python
"""Merge chronological per-initialization metric files without recomputation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    datasets = [np.load(path, allow_pickle=False) for path in args.input]
    for data in datasets[1:]:
        for key in ("lead_time", "variable"):
            if not np.array_equal(data[key], datasets[0][key]):
                raise RuntimeError(f"metric coordinate mismatch: {key}")
    init_time = np.concatenate([data["init_time"] for data in datasets])
    if np.any(init_time[1:] <= init_time[:-1]):
        raise RuntimeError("metric inputs are not strictly chronological")
    keys = ["mse", "mae", "bias"]
    optional = [
        "eligible_memory_count",
        "calibration_gate",
        "finite_ratio",
        "acc_numerator",
        "acc_prediction_energy",
        "acc_target_energy",
    ]
    payload = {
        "init_time": init_time,
        "lead_time": datasets[0]["lead_time"],
        "variable": datasets[0]["variable"],
    }
    for key in keys + optional:
        if all(key in data.files for data in datasets):
            payload[key] = np.concatenate([data[key] for data in datasets], axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
