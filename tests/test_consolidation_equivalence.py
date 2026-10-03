"""src/spherettc.py and src/spheredyn.py reproduce the archived originals exactly.

On 2026-10-03 SphereTTC and SphereDyn were consolidated into one file each.  The
superseded originals are archived byte-identically in
archive/pre_consolidation_20261003/.  These tests load the originals from the
archive and require bitwise-identical results.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from src.spheredyn import SphereDyn
from src.spherettc import (
    SphereTTC,
    SphereTTCCalibrator,
    SphereTTCConfig,
    apply_constrained_combination,
    fit_constrained_combination,
)
from src.ttc.metrics import latitude_weights, weighted_mean_spatial

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/pre_consolidation_20261003"
CHECKPOINT = ROOT / (
    "artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/"
    "spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt"
)
pytestmark = pytest.mark.skipif(not ARCHIVE.is_dir(), reason="archived originals not present")

SMALL = dict(
    image_size=(8, 16), width=16, local_layers=1, spectral_rank=4, spectral_lmax=6,
    spectral_bands=2, observation_width=16, observation_layers=1, observation_rank=4,
    observation_lmax=6, observation_bands=2, multiscale_width=16, multiscale_layers=1,
)


def _original_spheredyn_class():
    name = "_original_spheredyn_baselines"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ARCHIVE / "src/baselines")]
        sys.modules[name] = package
    return importlib.import_module(f"{name}.spheredyn_v9").SphereDynV9Forecast


def _original_module(relative: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ARCHIVE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


def _original_functions(relative: str, names: set[str]) -> dict:
    """Execute only the named top-level definitions of an archived script."""
    tree = ast.parse((ARCHIVE / relative).read_text(encoding="utf-8"))
    body = [
        node for node in tree.body
        if getattr(node, "name", None) in names
        or (isinstance(node, ast.Assign) and node.targets[0].id in names)
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(relative), "exec"), namespace)
    return namespace


# ----------------------------------------------------------------------------- SphereDyn
@pytest.mark.parametrize(
    "modes",
    [
        dict(multiscale_mode="learned"),
        dict(multiscale_mode="null_control"),
        dict(observation_mode="null_control", integration_mode="frozen_tendency"),
    ],
)
def test_spheredyn_small_model_matches_original_bitwise(modes):
    torch.manual_seed(0)
    original = _original_spheredyn_class()(54, 54, 2, **SMALL, **modes).eval()
    torch.manual_seed(0)
    merged = SphereDyn(54, 54, 2, **SMALL, **modes).eval()
    state = original.state_dict()
    assert list(state) == list(merged.state_dict())
    assert all(torch.equal(state[k], merged.state_dict()[k]) for k in state)  # same random init
    generator = torch.Generator().manual_seed(1)
    for value in state.values():
        if value.is_floating_point():
            value.copy_(0.1 * torch.randn(value.shape, generator=generator))
    merged.load_state_dict(state, strict=True)
    history, seasonal = torch.randn(2, 3, 54, 8, 16), torch.randn(2, 4)
    with torch.no_grad():
        assert torch.equal(original(history, seasonal), merged(history, seasonal))


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="SphereDyn checkpoint not present")
def test_spheredyn_checkpoint_forward_matches_original_bitwise():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    kwargs = dict(observation_width=192, observation_layers=6, observation_rank=80,
                  observation_lmax=80, observation_bands=10)
    original = _original_spheredyn_class()(54, 54, 5, **kwargs).eval()
    merged = SphereDyn(54, 54, 5).eval()
    original.load_state_dict(checkpoint["state_dict"], strict=True)
    merged.load_state_dict(checkpoint["state_dict"], strict=True)
    generator = torch.Generator().manual_seed(2)
    history = torch.randn(1, 3, 54, 121, 240, generator=generator)
    seasonal = torch.tensor([[0.5, 0.8660254, 0.8660254, -0.5]])
    with torch.no_grad():
        assert torch.equal(original(history, seasonal), merged(history, seasonal))


# ----------------------------------------------------------------------------- SphereTTC, Mode A
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


# ----------------------------------------------------------------------------- SphereTTC, Mode B
MODE_B_NAMES = {"RIDGE_RATIOS", "SPATIAL_SHRINKAGES", "_ridge_weights", "_energy",
                "_candidate_weights", "_feature_statistics", "_fit_backbone"}


def _statistics(rng, days, leads=2, variables=3, references=5, height=6, width=8):
    errors = rng.normal(size=(days, leads, variables, references + 1, height, width))
    errors[..., :references, :, :] += 0.5 * errors[..., -1:, :, :]  # correlated providers
    latitude = np.cos(np.deg2rad(np.linspace(80, -80, height)))
    flat = errors * np.sqrt(latitude)[None, None, None, None, :, None]
    matrix = flat.transpose(1, 2, 0, 4, 5, 3).reshape(leads, variables, -1, references + 1)
    cross = np.einsum("lvsm,lvsn->lvmn", matrix, matrix)
    sums = np.einsum("blvmxy,x->lvm", errors, latitude)
    return latitude, {
        "days": np.asarray(days),
        "count": np.asarray(float(days) * latitude.sum() * width),
        "reference_cross": cross[..., :references, :references],
        "reference_sum": sums[..., :references],
        "reference_map_sum": errors[..., :references, :, :].sum(axis=0).transpose(0, 1, 3, 4, 2),
        "raw_reference_cross": cross[None, ..., references, :references],
        "raw_sse": cross[None, ..., references, references],
        "raw_sum": sums[None, ..., references],
        "raw_map_sum": errors[..., references, :, :].sum(axis=0)[None],
    }


def test_mode_b_fit_matches_original_exactly():
    original = _original_functions("scripts/spheredyn/evaluate_goal_reference_ensemble.py", MODE_B_NAMES)
    rng = np.random.default_rng(4)
    latitude, train = _statistics(rng, 40)
    _, validation = _statistics(rng, 20)
    for minimum in (0.5, 0.0):
        old = original["_fit_backbone"](0, train, validation, latitude, minimum)
        new = fit_constrained_combination(train, validation, latitude, minimum)
        assert old.keys() == new.keys()
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])


def test_mode_b_apply_matches_main_run_formula():
    rng = np.random.default_rng(5)
    fitted = {
        "weights": rng.dirichlet(np.ones(6), size=(2, 3)),
        "bias": rng.normal(size=(2, 3)),
        "spatial_bias": rng.normal(size=(2, 3, 6, 8)),
    }
    raw = rng.normal(size=(4, 2, 3, 6, 8)).astype(np.float32)
    references = [rng.normal(size=raw.shape).astype(np.float32) for _ in range(5)]
    # verbatim arithmetic of scripts/spheredyn/run_main_spherettc_v2.py
    weights = torch.as_tensor(fitted["weights"], dtype=torch.float64)
    bias = torch.as_tensor(fitted["bias"], dtype=torch.float64)
    spatial_bias = torch.as_tensor(fitted["spatial_bias"], dtype=torch.float64)
    raw_t = torch.as_tensor(raw, dtype=torch.float64)
    expected = bias[None, :, :, None, None] - spatial_bias[None] + raw_t * weights[None, :, :, -1, None, None]
    for index, prediction in enumerate(references):
        expected = expected + torch.as_tensor(prediction, dtype=torch.float64) * weights[None, :, :, index, None, None]
    assert torch.equal(apply_constrained_combination(raw, references, fitted), expected)
