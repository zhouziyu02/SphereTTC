#!/usr/bin/env python
"""Freeze the simple global SphereTTC primary using 2017 metrics only."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import search_schedule as search


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "FROZEN_PRIMARY_GLOBAL_2017.json"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite frozen primary: {OUTPUT}")
    design = json.loads(search.DESIGN.read_text())
    rule = design["primary_global_selection"]
    search_start, search_stop = design["search_initialization_slice"]
    holdout_start, holdout_stop = design["internal_holdout_slice"]
    raw = search.load(ROOT / "metrics/validation/raw.npz")
    eligible = []
    evaluated = []
    for item in search.candidates():
        candidate_id = item["id"]
        calibrated = search.load(ROOT / f"metrics/validation/{candidate_id}.npz")
        first = search.summarize(raw, calibrated, search_start, search_stop)
        second = search.summarize(raw, calibrated, holdout_start, holdout_stop)
        passed = all(
            result["mean_relative_rmse_gain"]
            >= rule["minimum_mean_relative_rmse_gain_each_window"]
            and result["negative_rmse_cell_fraction"]
            <= rule["maximum_negative_rmse_cell_fraction_each_window"]
            and result["mean_acc_delta"] >= rule["minimum_mean_acc_delta_each_window"]
            for result in (first, second)
        )
        row = {
            "candidate_id": candidate_id,
            "params": {key: value for key, value in item.items() if key != "id"},
            "eligible": passed,
            "search_summary": first,
            "internal_holdout_summary": second,
            "worst_window_mean_relative_rmse_gain": min(
                first["mean_relative_rmse_gain"], second["mean_relative_rmse_gain"]
            ),
            "worst_window_mean_acc_delta": min(
                first["mean_acc_delta"], second["mean_acc_delta"]
            ),
        }
        evaluated.append(row)
        if passed:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no global candidate satisfies the frozen selection rule")
    selected = max(
        eligible,
        key=lambda row: (
            row["worst_window_mean_relative_rmse_gain"],
            row["worst_window_mean_acc_delta"],
            row["candidate_id"],
        ),
    )
    payload = {
        "status": "FROZEN_BEFORE_2020_METRICS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prospective_2020_metrics_touched": False,
        "selection_data": "2017 only",
        "selection_rule": rule,
        "selected_candidate_id": selected["candidate_id"],
        "selected_params": selected["params"],
        "search_summary": selected["search_summary"],
        "internal_holdout_summary": selected["internal_holdout_summary"],
        "eligible_candidate_ids": sorted(row["candidate_id"] for row in eligible),
        "candidate_count": len(evaluated),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "PRIMARY_GLOBAL_FROZEN",
        selected["candidate_id"],
        f"search_gain={100 * selected['search_summary']['mean_relative_rmse_gain']:.4f}%",
        f"holdout_gain={100 * selected['internal_holdout_summary']['mean_relative_rmse_gain']:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
