from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _metric(path: Path, mse: float) -> None:
    np.savez_compressed(
        path,
        mse=np.full((4, 2, 1), mse, dtype=np.float64),
        init_time=np.arange(4),
        lead_time=np.asarray([24, 72]),
        variable=np.asarray(["z500"]),
    )


def test_main_goal_can_pass_while_graphcast_is_better(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    final = tmp_path / "final.npz"
    graphcast = tmp_path / "graphcast.npz"
    stats = tmp_path / "stats.json"
    output = tmp_path / "gate.json"
    _metric(raw, 1.0)
    _metric(final, 0.81)
    _metric(graphcast, 0.25)
    stats.write_text(json.dumps({"variables": ["z500"], "std": [1.0]}))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/spheredyn/assess_main_prediction_goal.py"),
            "--raw",
            str(raw),
            "--final",
            str(final),
            "--graphcast",
            str(graphcast),
            "--stats",
            str(stats),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "MAIN_PREDICTION_GOAL_ACHIEVED"
    assert np.isclose(result["gain_percent"], 10.0)
    assert result["graphcast_macro_nrmse"] < result["final_spheredyn_plus_spherettc_macro_nrmse"]
    assert result["graphcast_hard_gate"] is False


def test_current_protocol_requires_only_one_seed() -> None:
    protocol = json.loads(
        (
            ROOT
            / "artifacts/spheredyn_spherettc_open_goal_20260801/PROTOCOL_V2.json"
        ).read_text()
    )
    assert protocol["main_experiment"]["seed_count"] == 1
    assert protocol["graphcast_policy"]["hard_gate"] is False
    assert protocol["explicitly_not_required"]["ablation_experiments"] is True
