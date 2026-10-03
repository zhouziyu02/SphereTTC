#!/usr/bin/env python
"""Summarize every 2017 SphereTTC candidate on both chronological windows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import search_schedule as search


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    design = json.loads(search.DESIGN.read_text())
    windows = {
        "search": design["search_initialization_slice"],
        "internal_holdout": design["internal_holdout_slice"],
        "full_development": [
            design["search_initialization_slice"][0],
            design["internal_holdout_slice"][1],
        ],
    }
    raw = search.load(ROOT / "metrics/validation/raw.npz")
    rows = []
    for item in search.candidates():
        calibrated = search.load(ROOT / f"metrics/validation/{item['id']}.npz")
        row = {"candidate_id": item["id"]}
        for label, (start, stop) in windows.items():
            result = search.summarize(raw, calibrated, start, stop)
            row[f"{label}_mean_relative_rmse_gain"] = result["mean_relative_rmse_gain"]
            row[f"{label}_negative_rmse_cells"] = result["negative_rmse_cells"]
            row[f"{label}_mean_acc_delta"] = result["mean_acc_delta"]
            row[f"{label}_negative_acc_cells"] = result["negative_acc_cells"]
        rows.append(row)
    rows.sort(
        key=lambda row: (
            min(
                row["search_mean_relative_rmse_gain"],
                row["internal_holdout_mean_relative_rmse_gain"],
            ),
            min(row["search_mean_acc_delta"], row["internal_holdout_mean_acc_delta"]),
        ),
        reverse=True,
    )
    output = ROOT / "VALIDATION_CANDIDATES_2017.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("VALIDATION_ANALYSIS_COMPLETE", output)
    for row in rows[:10]:
        print(
            row["candidate_id"],
            f"search_gain={100 * row['search_mean_relative_rmse_gain']:.4f}%",
            f"search_acc={row['search_mean_acc_delta']:+.8f}",
            f"holdout_gain={100 * row['internal_holdout_mean_relative_rmse_gain']:.4f}%",
            f"holdout_acc={row['internal_holdout_mean_acc_delta']:+.8f}",
        )


if __name__ == "__main__":
    main()
