import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def test_run_ttc_persists_daily_climatology_acc_components(tmp_path):
    init_time = np.asarray(["2018-02-27", "2018-02-28"], dtype="datetime64[ns]")
    lead_time = np.asarray([24], dtype=np.int32)
    valid_time = np.asarray(
        [["2018-02-28"], ["2018-03-01"]], dtype="datetime64[ns]"
    )
    lat = np.asarray([-30.0, 30.0], dtype=np.float64)
    lon = np.asarray([0.0, 180.0], dtype=np.float64)
    variable = np.asarray(["z500"])
    pred = np.asarray(
        [[[[[2.0, 2.0], [2.0, 2.0]]]], [[[[13.0, 13.0], [13.0, 13.0]]]]],
        dtype=np.float32,
    )
    target = np.asarray(
        [[[[[1.0, 1.0], [1.0, 1.0]]]], [[[[12.0, 12.0], [12.0, 12.0]]]]],
        dtype=np.float32,
    )
    cache = tmp_path / "cache.zarr"
    xr.Dataset(
        data_vars={
            "pred": (
                ("init_time", "lead_time", "variable", "lat", "lon"),
                pred,
            ),
            "target": (
                ("init_time", "lead_time", "variable", "lat", "lon"),
                target,
            ),
            "valid_time": (("init_time", "lead_time"), valid_time),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "variable": variable,
            "lat": lat,
            "lon": lon,
        },
    ).to_zarr(cache, mode="w", consolidated=True)

    climatology = np.zeros((366, 1, 2, 2), dtype=np.float32)
    climatology[58] = 0.5
    climatology[60] = 10.0
    climatology_path = tmp_path / "daily_climatology.npz"
    np.savez_compressed(
        climatology_path,
        calendar_day=np.arange(1, 367, dtype=np.int16),
        climatology=climatology,
        variable=variable,
        lat=lat,
        lon=lon,
    )
    metrics_path = tmp_path / "metrics.npz"
    profile_path = tmp_path / "profile.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_ttc.py",
            "--config",
            "configs/unified_11model_ttc.yaml",
            "--model",
            "smoke",
            "--method",
            "raw",
            "--cache",
            str(cache),
            "--no-cache-output",
            "--metrics-output",
            str(metrics_path),
            "--profile-output",
            str(profile_path),
            "--acc-climatology",
            str(climatology_path),
            "--allow-cpu-metric-replay",
        ],
        cwd=ROOT,
        check=True,
    )

    with np.load(metrics_path, allow_pickle=False) as metrics:
        for key in (
            "acc_numerator",
            "acc_prediction_energy",
            "acc_target_energy",
        ):
            assert key in metrics
            assert np.isfinite(metrics[key]).all()
