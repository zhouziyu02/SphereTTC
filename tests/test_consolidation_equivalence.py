"""The consolidated SphereTTC preserves the original calibrator and runner.

The original calibration and memory modules are archived byte-identically in
archive/pre_consolidation_20261003/.  These tests require bitwise-identical
calibration and agreement between the online wrapper and the experiment runner.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from src.spherettc import (
    SphereTTC,
    SphereTTCCalibrator,
    SphereTTCConfig,
)
from src.ttc.metrics import latitude_weights, weighted_mean_spatial

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/pre_consolidation_20261003"
pytestmark = pytest.mark.skipif(not ARCHIVE.is_dir(), reason="archived originals not present")


def _original_module(relative: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ARCHIVE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("strength, lmax", [(1.0, 8), (0.8, 6)])
def test_sphere_ttc_calibrator_matches_original_bitwise(strength, lmax):
    original = _original_module("src/ttc/sphere.py", "_original_sphere")
    values = dict(lmax=lmax, mmax=lmax, min_memory=8, min_holdout=4, strength=strength)
    old = original.SphereTTCCalibrator(17, 32, torch.device("cpu"), original.SphereTTCConfig(**values))
    new = SphereTTCCalibrator(17, 32, torch.device("cpu"), SphereTTCConfig(**values))
    generator = torch.Generator().manual_seed(3)
    bias = torch.randn(3, 17, 32, generator=generator)
    truth = torch.randn(40, 3, 17, 32, generator=generator)
    pred = truth + bias + 0.3 * torch.randn(40, 3, 17, 32, generator=generator)
    current = torch.randn(3, 17, 32, generator=generator) + bias
    current[0, 0, 0] = float("nan")
    a = old.calibrate(current, old.encode(pred), old.encode(truth))
    b = new.calibrate(current, new.encode(pred), new.encode(truth))
    assert torch.equal(a[1], b[1])
    assert torch.equal(torch.nan_to_num(a[0], nan=-1e30), torch.nan_to_num(b[0], nan=-1e30))


def test_online_sphere_ttc_matches_run_ttc(tmp_path):
    rng = np.random.default_rng(0)
    n_init, n_lat, n_lon = 60, 17, 32
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
        coords={"init_time": init.astype("datetime64[ns]"), "lead_time": leads,
                "variable": ["z500", "t850"], "lat": lat, "lon": lon},
    ).to_zarr(cache, mode="w", consolidated=True)
    climatology = tmp_path / "climatology.npz"
    np.savez_compressed(
        climatology, calendar_day=np.arange(1, 367, dtype=np.int16),
        climatology=np.zeros((366, 2, n_lat, n_lon), dtype=np.float32),
        variable=np.asarray(["z500", "t850"]), lat=lat, lon=lon,
    )
    subprocess.run(
        [sys.executable, "scripts/run_ttc.py", "--config", "configs/unified_11model_ttc.yaml",
         "--model", "smoke", "--method", "sphere_ttc", "--cache", str(cache), "--no-cache-output",
         "--metrics-output", str(tmp_path / "m.npz"), "--profile-output", str(tmp_path / "p.json"),
         "--acc-climatology", str(climatology), "--allow-cpu-metric-replay",
         "--lmax", "8", "--mmax", "8", "--memory-size", "40", "--min-memory", "24"],
        cwd=ROOT, check=True, capture_output=True,
    )
    with np.load(tmp_path / "m.npz") as metrics:
        runner_mse, runner_gate = metrics["mse"], metrics["calibration_gate"]
        runner_count = metrics["eligible_memory_count"]

    ttc = SphereTTC(n_lat, n_lon, leads, SphereTTCConfig(lmax=8, mmax=8, min_memory=24), memory_size=40)
    weights = latitude_weights(lat, torch.device("cpu"))
    for i, t in enumerate(init.astype("datetime64[ns]")):
        calibrated, gate, count = ttc.calibrate(t, pred[i])
        target = torch.as_tensor(truth[i])
        for lead in range(len(leads)):  # same per-lead metric as scripts/run_ttc.py
            mse = weighted_mean_spatial((calibrated[lead] - target[lead]).square(), weights).numpy()
            np.testing.assert_array_equal(mse, runner_mse[i, lead].astype(np.float32))
        np.testing.assert_array_equal(gate, runner_gate[i])
        assert count == list(runner_count[i])
        ttc.update(t, pred[i], truth[i])
    assert runner_gate[-1].mean() > 0.5  # the comparison exercised active calibration
