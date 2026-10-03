"""End-to-end smoke test of the final SphereTTC through scripts/run_ttc.py.

A synthetic forecast cache carries a stable, planetary-scale bias.  SphereTTC
must learn it causally from delayed memory and remove most of it, while the
raw method leaves it untouched.  Runs on CPU in a few seconds.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def _run(method, cache, climatology, out):
    subprocess.run(
        [
            sys.executable, "scripts/run_ttc.py",
            "--config", "configs/unified_11model_ttc.yaml",
            "--model", "smoke", "--method", method,
            "--cache", str(cache), "--no-cache-output",
            "--metrics-output", str(out / f"{method}.npz"),
            "--profile-output", str(out / f"{method}.json"),
            "--acc-climatology", str(climatology),
            "--allow-cpu-metric-replay",
            "--lmax", "8", "--mmax", "8", "--memory-size", "48",
        ],
        cwd=ROOT, check=True, capture_output=True,
    )
    with np.load(out / f"{method}.npz") as metrics:
        return metrics["mse"].astype(np.float64), metrics["calibration_gate"]


def test_sphere_ttc_removes_coherent_bias_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n_init, n_lat, n_lon = 72, 17, 32
    init = np.datetime64("2018-01-01", "D") + np.arange(n_init).astype("timedelta64[D]")
    leads = np.asarray([24, 72], dtype=np.int32)
    valid = init[:, None] + (leads // 24).astype("timedelta64[D]")[None]
    lat = np.linspace(90.0, -90.0, n_lat)
    lon = np.arange(n_lon) * 360.0 / n_lon
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    bias = (2.0 * np.cos(phi) * np.cos(lam) + np.sin(phi)).astype(np.float32)
    truth = rng.normal(size=(n_init, 2, 2, n_lat, n_lon)).astype(np.float32)
    pred = truth + bias + 0.3 * rng.normal(size=truth.shape).astype(np.float32)
    cache = tmp_path / "cache.zarr"
    xr.Dataset(
        {
            "pred": (("init_time", "lead_time", "variable", "lat", "lon"), pred),
            "target": (("init_time", "lead_time", "variable", "lat", "lon"), truth),
            "valid_time": (("init_time", "lead_time"), valid.astype("datetime64[ns]")),
        },
        coords={
            "init_time": init.astype("datetime64[ns]"), "lead_time": leads,
            "variable": ["z500", "t850"], "lat": lat, "lon": lon,
        },
    ).to_zarr(cache, mode="w", consolidated=True)
    climatology = tmp_path / "climatology.npz"
    np.savez_compressed(
        climatology,
        calendar_day=np.arange(1, 367, dtype=np.int16),
        climatology=np.zeros((366, 2, n_lat, n_lon), dtype=np.float32),
        variable=np.asarray(["z500", "t850"]), lat=lat, lon=lon,
    )
    raw_mse, raw_gate = _run("raw", cache, climatology, tmp_path)
    ttc_mse, ttc_gate = _run("sphere_ttc", cache, climatology, tmp_path)
    warm = slice(40, None)  # after memory has filled (min_memory + delay)
    assert np.all(raw_gate == 0)
    assert ttc_gate[warm].mean() > 0.5
    assert ttc_mse[warm].mean() < 0.25 * raw_mse[warm].mean()
    # Causality: before any verified pair exists the forecast is untouched.
    np.testing.assert_allclose(ttc_mse[0], raw_mse[0], rtol=1e-6)
