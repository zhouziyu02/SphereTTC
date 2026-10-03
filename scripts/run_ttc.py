#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import dask.array as da
import numpy as np
import torch
import zarr

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.spherettc import EligibleMemory, assert_no_future_targets
from src.ttc.comparators.geo import apply_geo_ttc
from src.ttc.metrics import latitude_weights, weighted_mean_spatial
from src.spherettc import SphereTTCConfig, SphereTTCCalibrator
from src.ttc.legacy.sphere_v2 import SphereTTCv2Calibrator, SphereTTCv2Config
from src.ttc.legacy.sphere_v3 import SphereTTCv3Calibrator, SphereTTCv3Config
from src.ttc.legacy.sphere_v4 import SphereTTCv4Calibrator, SphereTTCv4Config
from src.ttc.legacy.sphere_v5 import SphereTTCv5Calibrator, SphereTTCv5Config
from src.ttc.legacy.sphere_v6 import SphereTTCv6Calibrator, SphereTTCv6Config
from src.ttc.legacy.sphere_v7 import SphereTTCv7Calibrator, SphereTTCv7Config
from src.ttc.comparators.st_ttc import STTTCSDCalibrator
from src.ttc.comparators.spectral import apply_zonal_spectral
from src.utils.config import load_config
from src.utils.device import get_device_info, require_cuda
from src.utils.climatology import calendar_day_index


def _prepare_output(ds, out: Path, method: str, include_target: bool = True):
    import xarray as xr

    if out.exists():
        shutil.rmtree(out)
    shape = ds["pred"].shape
    chunks = (1, 1, shape[2], shape[3], shape[4])
    data_vars = {
            "pred": (ds["pred"].dims, da.empty(shape, chunks=chunks, dtype="float32")),
            "valid_time": (ds["valid_time"].dims, ds["valid_time"].values),
            "eligible_memory_count": (
                ("init_time", "lead_time"),
                da.zeros((shape[0], shape[1]), chunks=(min(128, shape[0]), shape[1]), dtype="int32"),
            ),
            "calibration_gate": (
                ("init_time", "lead_time", "variable"),
                da.zeros(
                    (shape[0], shape[1], shape[2]),
                    chunks=(min(128, shape[0]), shape[1], shape[2]),
                    dtype="float32",
                ),
            ),
        }
    if include_target:
        data_vars["target"] = (ds["target"].dims, da.empty(shape, chunks=chunks, dtype="float32"))
    template = xr.Dataset(
        data_vars=data_vars,
        coords={name: ds.coords[name].values for name in ds.coords},
        attrs=dict(ds.attrs),
    )
    template.attrs["method"] = method
    template.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
    template.attrs["device_info"] = str(get_device_info())
    template.to_zarr(out, mode="w", consolidated=False, compute=False)
    return zarr.open_group(str(out), mode="a")


def _as_torch(array, device):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)


# The causal memory-admission rule lives in src/spherettc.py (single source of truth).
from src.spherettc import eligible_from_buffer as _eligible_from_buffer  # noqa: E402


def _finite_residual(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    residual = target - pred
    return torch.where(torch.isfinite(residual), residual, torch.zeros_like(residual))


def _finite_memory_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(pred).logical_and(torch.isfinite(target))
    target_safe = torch.nan_to_num(target)
    pred_safe = torch.where(finite, pred, target_safe)
    return pred_safe, target_safe


def _restrict_warmup_period(
    dataset: xr.Dataset,
    end_exclusive: str | None,
) -> xr.Dataset:
    """Restrict a reusable multi-year cache to a causal warmup prefix."""
    if end_exclusive is None:
        return dataset
    cutoff = np.datetime64(end_exclusive, "ns")
    times = np.asarray(dataset["init_time"].values).astype("datetime64[ns]")
    keep = times < cutoff
    if not np.any(keep):
        raise RuntimeError(
            "warmup cache has no initializations before "
            f"--warmup-end-exclusive={end_exclusive}"
        )
    return dataset.isel(init_time=np.flatnonzero(keep))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["raw", "static_bias", "ema_bias", "affine_ttc", "zonal_spectral_ttc", "geo_ttc", "st_ttc", "sphere_ttc", "sphere_ttc_v2", "sphere_ttc_v3", "sphere_ttc_v4", "sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"],
    )
    parser.add_argument("--adapt-steps", type=int, default=None)
    parser.add_argument("--memory-size", type=int, default=None)
    parser.add_argument("--n-modes", type=int, default=None)
    parser.add_argument("--snr-lambda", type=float, default=4.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--min-memory", type=int, default=8)
    parser.add_argument("--sample-tau", type=float, default=16.0)
    parser.add_argument("--lmax", type=int, default=16)
    parser.add_argument("--mmax", type=int, default=16)
    parser.add_argument("--half-life", type=float, default=32.0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--min-holdout", type=int, default=8)
    parser.add_argument("--confidence-z", type=float, default=0.5)
    parser.add_argument("--robust-quantile-scale", type=float, default=3.0)
    parser.add_argument("--max-scale-delta", type=float, default=0.5)
    parser.add_argument("--taper-power", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--risk-fraction", type=float, default=0.5)
    parser.add_argument("--n-bands", type=int, default=4)
    parser.add_argument("--gate-ridge", type=float, default=0.05)
    parser.add_argument("--max-component-blend", type=float, default=1.0)
    parser.add_argument("--risk-floor", type=float, default=0.01)
    parser.add_argument("--hierarchy-weight", type=float, default=0.25)
    parser.add_argument("--max-blend", type=float, default=1.0)
    parser.add_argument("--coefficient-fraction", type=float, default=0.25)
    parser.add_argument("--short-memory-size", type=int, default=32)
    parser.add_argument("--medium-memory-size", type=int, default=96)
    parser.add_argument("--short-half-life", type=float, default=12.0)
    parser.add_argument("--medium-half-life", type=float, default=48.0)
    parser.add_argument("--long-half-life", type=float, default=64.0)
    parser.add_argument("--mixture-ridge", type=float, default=0.05)
    parser.add_argument("--protection-tolerance", type=float, default=0.02)
    parser.add_argument("--bias-anchor-half-life", type=float, default=64.0)
    parser.add_argument(
        "--use-bias-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-blocks", type=int, default=3)
    parser.add_argument("--worst-block-weight", type=float, default=0.5)
    parser.add_argument(
        "--use-seasonal-anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seasonal-half-life", type=float, default=730.0)
    parser.add_argument("--seasonal-ridge", type=float, default=0.1)
    parser.add_argument("--seasonal-harmonic-order", type=int, default=2)
    parser.add_argument("--annual-period-days", type=float, default=365.2425)
    parser.add_argument(
        "--expert-mode",
        choices=["mixture", "short", "long"],
        default="mixture",
    )
    parser.add_argument("--st-ttc-groups", type=int, default=3)
    parser.add_argument("--st-ttc-lr", type=float, default=1e-4)
    parser.add_argument("--st-ttc-updates", type=int, default=1)
    parser.add_argument(
        "--normalization-stats",
        default="artifacts/unified_11model/s2s_daily_54var_stats.json",
    )
    parser.add_argument("--params-json", default=None, help="Per-model frozen SphereTTC hyperparameters.")
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument(
        "--lead-time",
        type=int,
        default=None,
        help="Evaluate one lead only; used for process-parallel CPU metric replay.",
    )
    parser.add_argument("--init-times-from", default=None)
    parser.add_argument(
        "--init-start",
        default=None,
        help=(
            "Optional inclusive initialization-date embargo boundary applied "
            "after --init-times-from."
        ),
    )
    parser.add_argument(
        "--init-end-inclusive",
        default=None,
        help=(
            "Optional inclusive final initialization date applied after "
            "--init-times-from."
        ),
    )
    parser.add_argument(
        "--warmup-cache",
        nargs="+",
        default=None,
        help="One or more chronological earlier caches used only to seed causal memory.",
    )
    parser.add_argument(
        "--trim-warmup-cache",
        action="store_true",
        help=(
            "Read only the chronological tail that can survive the bounded "
            "memory buffer. Safe for stateless bounded-buffer calibrators."
        ),
    )
    parser.add_argument(
        "--warmup-end-exclusive",
        default=None,
        help=(
            "Restrict every warmup cache to initializations before this date. "
            "This permits causal reuse of a multi-year cache."
        ),
    )
    parser.add_argument(
        "--warmup-truth-root",
        default=None,
        help=(
            "Canonical truth root for warmup caches when it differs from "
            "--truth-root (for example, an isolated prospective test root)."
        ),
    )
    parser.add_argument("--cache", default=None, help="Source forecast cache; defaults to the canonical cache for --model.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--omit-target", action="store_true", help="Do not duplicate canonical truth in the output cache.")
    parser.add_argument("--no-cache-output", action="store_true", help="Compute TTC metrics without materializing a prediction cache.")
    parser.add_argument("--metrics-output", default=None, help="Optional compact per-initialization metric NPZ.")
    parser.add_argument(
        "--profile-output",
        default=None,
        help="Optional explicit profile JSON path for isolated metric replays.",
    )
    parser.add_argument(
        "--acc-climatology",
        default=None,
        help=(
            "Fixed NPZ climatology built by build_publication_acc_reference.py. "
            "When supplied, save per-initialization ACC covariance components."
        ),
    )
    parser.add_argument("--truth-root", default=None, help="Override cache targets with canonical daily S2S truth.")
    parser.add_argument("--truth-workers", type=int, default=8)
    parser.add_argument(
        "--allow-cpu-metric-replay",
        action="store_true",
        help=(
            "Allow CPU only for no-cache metric replay. This is intended for "
            "post-hoc metric recovery and never permits formal training/inference."
        ),
    )
    parser.add_argument(
        "--cpu-lead-workers",
        type=int,
        default=1,
        help="Parallel independent lead calibrations during CPU metric replay.",
    )
    args = parser.parse_args()
    import xarray as xr

    metrics_lock = None
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_lock = (metrics_path.parent / f"{metrics_path.name}.lock").open("a+")
        try:
            fcntl.flock(metrics_lock.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.ENOSYS, errno.EOPNOTSUPP):
                raise
            # Some distributed filesystems do not implement flock(). The
            # shape/finite validation below still makes sequential retries
            # idempotent; orchestration must avoid concurrent writers there.
            metrics_lock.close()
            metrics_lock = None

    cfg = load_config(args.config)
    if args.params_json:
        tuned = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        tuned = tuned.get("best_params", tuned)
        for name in (
            "memory_size",
            "lmax",
            "mmax",
            "half_life",
            "ridge",
            "strength",
            "min_memory",
            "holdout_fraction",
            "min_holdout",
            "confidence_z",
            "robust_quantile_scale",
            "max_scale_delta",
            "taper_power",
            "validation_fraction",
            "risk_fraction",
            "n_bands",
            "gate_ridge",
            "max_component_blend",
            "risk_floor",
            "hierarchy_weight",
            "max_blend",
            "coefficient_fraction",
            "short_memory_size",
            "medium_memory_size",
            "short_half_life",
            "medium_half_life",
            "long_half_life",
            "mixture_ridge",
            "protection_tolerance",
            "bias_anchor_half_life",
            "use_bias_anchor",
            "validation_blocks",
            "worst_block_weight",
            "use_seasonal_anchor",
            "seasonal_half_life",
            "seasonal_ridge",
            "seasonal_harmonic_order",
            "annual_period_days",
            "expert_mode",
        ):
            if name in tuned:
                setattr(args, name, tuned[name])
    if args.allow_cpu_metric_replay:
        if not args.no_cache_output or not args.metrics_output or not args.acc_climatology:
            raise RuntimeError(
                "--allow-cpu-metric-replay requires --no-cache-output, "
                "--metrics-output and --acc-climatology"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = require_cuda(f"{args.method} TTC")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cache = Path(args.cache) if args.cache else Path(f"artifacts/caches/{args.model}/forecast_cache_2020_121x240.zarr")
    if args.model == "oneforecast" and not cache.exists():
        cache = Path("artifacts/caches/oneforecast/forecast_cache_2020_120x240.zarr")
    ds = xr.open_zarr(cache, consolidated=True)
    # Preserve the protocol-wide maximum lead for causal buffer retention even
    # when a metric replay selects one lead. This keeps single-lead replay
    # exactly aligned with the original five-lead run.
    protocol_max_lead_hours = max(float(value) for value in ds["lead_time"].values)
    if args.variables:
        available = [str(x) for x in ds["variable"].values]
        missing = [v for v in args.variables if v not in available]
        if missing:
            raise RuntimeError(f"{cache} is missing requested variables: {missing}")
        ds = ds.sel(variable=args.variables)
    if args.lead_time is not None:
        available_leads = [int(value) for value in ds["lead_time"].values]
        if args.lead_time not in available_leads:
            raise RuntimeError(f"cache is missing requested lead time: {args.lead_time}")
        ds = ds.sel(lead_time=[args.lead_time])
    if args.init_times_from:
        ref = xr.open_zarr(args.init_times_from, consolidated=True)
        ref_times = np.asarray(ref["init_time"].values).astype("datetime64[ns]")
        current_times = np.asarray(ds["init_time"].values).astype("datetime64[ns]")
        keep = ref_times[np.isin(ref_times, current_times)]
        if keep.size != ref_times.size:
            missing = ref_times[~np.isin(ref_times, current_times)]
            raise RuntimeError(f"{cache} is missing {missing.size} reference init times; first missing={missing[:3]}")
        ds = ds.sel(init_time=keep)
    if args.init_start or args.init_end_inclusive:
        times = np.asarray(ds["init_time"].values).astype("datetime64[ns]")
        keep = np.ones(times.shape, dtype=bool)
        if args.init_start:
            keep &= times >= np.datetime64(args.init_start, "ns")
        if args.init_end_inclusive:
            keep &= times <= np.datetime64(
                args.init_end_inclusive,
                "ns",
            )
        if not keep.any():
            raise RuntimeError(
                "initialization-date selection produced an empty stream"
            )
        selected_times = times[keep]
        if selected_times.size > 1 and not np.all(
            np.diff(selected_times) == np.timedelta64(1, "D")
        ):
            raise RuntimeError(
                "selected initialization stream is not complete daily data"
            )
        ds = ds.isel(init_time=np.flatnonzero(keep))
    init_times = ds["init_time"].values
    valid_times = ds["valid_time"].values
    memory_size = args.memory_size or int(cfg["ttc"]["memory_size"])
    shape = ds["pred"].shape
    n_init, n_lead, n_var, n_lat, n_lon = shape
    selected_variables = [str(x) for x in ds["variable"].values]
    lead_values = [int(x) for x in ds["lead_time"].values]
    if args.metrics_output and Path(args.metrics_output).is_file():
        metrics_path = Path(args.metrics_output)
        try:
            with np.load(metrics_path, allow_pickle=False) as existing:
                required_shapes = {
                    "init_time": (n_init,),
                    "lead_time": (n_lead,),
                    "variable": (n_var,),
                    "mse": (n_init, n_lead, n_var),
                    "mae": (n_init, n_lead, n_var),
                    "bias": (n_init, n_lead, n_var),
                    "eligible_memory_count": (n_init, n_lead),
                    "calibration_gate": (n_init, n_lead, n_var),
                }
                if args.acc_climatology:
                    required_shapes.update(
                        {
                            "acc_numerator": (n_init, n_lead, n_var),
                            "acc_prediction_energy": (n_init, n_lead, n_var),
                            "acc_target_energy": (n_init, n_lead, n_var),
                        }
                    )
                valid_existing = all(
                    key in existing and existing[key].shape == expected
                    for key, expected in required_shapes.items()
                ) and all(
                    np.isfinite(existing[key]).all()
                    for key in required_shapes
                    if key
                    in {
                        "mse",
                        "mae",
                        "bias",
                        "calibration_gate",
                        "acc_numerator",
                        "acc_prediction_energy",
                        "acc_target_energy",
                    }
                ) and (
                    np.array_equal(
                        existing["init_time"],
                        np.asarray(init_times)
                        .astype("datetime64[ns]")
                        .astype(np.int64),
                    )
                    and np.array_equal(
                        existing["lead_time"],
                        np.asarray(lead_values, dtype=np.int32),
                    )
                    and np.array_equal(
                        existing["variable"],
                        np.asarray(selected_variables),
                    )
                )
            if valid_existing:
                print(f"REUSING complete metrics {metrics_path}", flush=True)
                return
        except (OSError, ValueError, KeyError):
            pass
    suffix = "120x240" if cache.name.endswith("120x240.zarr") else "121x240"
    out = Path(args.output) if args.output else Path(f"artifacts/caches/{args.model}/calibrated_{args.method}_2020_{suffix}.zarr")
    out_group = (
        None
        if args.no_cache_output
        else _prepare_output(ds, out, args.method, include_target=not args.omit_target)
    )
    if args.no_cache_output and not args.metrics_output:
        raise RuntimeError("--no-cache-output requires --metrics-output")
    metric_mse = np.empty((n_init, n_lead, n_var), dtype=np.float64) if args.metrics_output else None
    metric_mae = np.empty_like(metric_mse) if args.metrics_output else None
    metric_bias = np.empty_like(metric_mse) if args.metrics_output else None
    metric_weights = latitude_weights(ds["lat"].values, device) if args.metrics_output else None
    metric_acc_numerator = None
    metric_acc_prediction_energy = None
    metric_acc_target_energy = None
    metric_acc_climatology = None
    metric_acc_climatology_by_day = None
    metric_acc_day_indices = None
    if args.acc_climatology:
        if not args.metrics_output:
            raise RuntimeError("--acc-climatology requires --metrics-output")
        with np.load(args.acc_climatology, allow_pickle=False) as acc_reference:
            for key, current in (
                ("variable", np.asarray(selected_variables)),
                ("lat", np.asarray(ds["lat"].values, dtype=np.float64)),
                ("lon", np.asarray(ds["lon"].values, dtype=np.float64)),
            ):
                if key not in acc_reference or not np.array_equal(acc_reference[key], current):
                    raise RuntimeError(f"ACC climatology coordinate mismatch: {key}")
            if "calendar_day" in acc_reference:
                calendar_days = np.asarray(acc_reference["calendar_day"], dtype=np.int16)
                if not np.array_equal(calendar_days, np.arange(1, 367, dtype=np.int16)):
                    raise RuntimeError("ACC daily climatology calendar must contain days 1..366")
                climatology = np.asarray(acc_reference["climatology"], dtype=np.float32)
                expected_shape = (366, n_var, n_lat, n_lon)
                if climatology.shape != expected_shape:
                    raise RuntimeError(
                        f"ACC daily climatology shape mismatch: {climatology.shape} "
                        f"!= {expected_shape}"
                    )
                metric_acc_climatology_by_day = climatology
                metric_acc_day_indices = calendar_day_index(valid_times)
            else:
                reference_leads = np.asarray(acc_reference["lead_time"], dtype=np.int32)
                lead_indices = [
                    int(np.where(reference_leads == lead)[0][0]) for lead in lead_values
                ]
                climatology = np.asarray(
                    acc_reference["climatology"][lead_indices], dtype=np.float32
                )
                expected_shape = (n_lead, n_var, n_lat, n_lon)
                if climatology.shape != expected_shape:
                    raise RuntimeError(
                        f"ACC climatology shape mismatch: {climatology.shape} "
                        f"!= {expected_shape}"
                    )
                metric_acc_climatology = _as_torch(climatology, device)
        if not np.isfinite(climatology).all():
            raise RuntimeError("ACC climatology contains non-finite values")
        metric_acc_numerator = np.empty((n_init, n_lead, n_var), dtype=np.float64)
        metric_acc_prediction_energy = np.empty_like(metric_acc_numerator)
        metric_acc_target_energy = np.empty_like(metric_acc_numerator)

    ema_state = None
    opt = None
    scale = bias = gate = None
    weights = None
    k = None
    sphere = None
    st_ttc = None
    st_optimizer = None
    st_buffer = []
    st_last_update = [None]
    st_mean = st_std = None
    adapt_steps = args.adapt_steps or int(cfg["ttc"]["adapt_steps"])
    init_step_hours = max(1.0, float((init_times[1] - init_times[0]) / np.timedelta64(1, "h"))) if n_init > 1 else 12.0
    max_lead_hours = max(
        protocol_max_lead_hours,
        max(
            float((valid_times[i, l] - init_times[i]) / np.timedelta64(1, "h"))
            for i in range(min(n_init, 2))
            for l in range(n_lead)
        ),
    )
    keep_size = int(memory_size or 32) + int(np.ceil(max_lead_hours / init_step_hours)) + 4
    residual_buffers = [[] for _ in range(n_lead)]
    sample_buffers = [[] for _ in range(n_lead)]
    coefficient_buffers = [[] for _ in range(n_lead)]
    if args.method == "ema_bias":
        ema_state = torch.zeros((n_lead, n_var, n_lat, n_lon), dtype=torch.float32, device=device)
    elif args.method == "affine_ttc":
        scale = torch.zeros((n_lead, n_var, 1, 1), device=device, requires_grad=True)
        bias = torch.zeros((n_lead, n_var, 1, 1), device=device, requires_grad=True)
        opt = torch.optim.Adam([scale, bias], lr=float(cfg["ttc"]["lr"]))
        weights = latitude_weights(ds["lat"].values, device)
    elif args.method == "zonal_spectral_ttc":
        k = min(int(args.n_modes or cfg["ttc"]["n_modes"]), n_lon // 2 + 1)
        gate = torch.ones((n_lead, n_var, 1, k), dtype=torch.complex64, device=device, requires_grad=True)
        bias = torch.zeros((n_lead, n_var, 1, k), dtype=torch.complex64, device=device, requires_grad=True)
        opt = torch.optim.Adam([gate, bias], lr=float(cfg["ttc"]["lr"]))
        weights = latitude_weights(ds["lat"].values, device)
    elif args.method == "geo_ttc":
        k = min(int(args.n_modes or cfg["ttc"]["n_modes"]), n_lon // 2 + 1)
    elif args.method in {
        "sphere_ttc",
        "sphere_ttc_v2",
        "sphere_ttc_v3",
        "sphere_ttc_v4",
        "sphere_ttc_v5",
        "sphere_ttc_v6",
        "sphere_ttc_v7",
    }:
        if args.method == "sphere_ttc_v7":
            sphere = SphereTTCv7Calibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCv7Config(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    n_bands=args.n_bands,
                    memory_size=memory_size,
                    min_memory=args.min_memory,
                    short_memory_size=args.short_memory_size,
                    medium_memory_size=args.medium_memory_size,
                    short_half_life=args.short_half_life,
                    medium_half_life=args.medium_half_life,
                    long_half_life=args.long_half_life,
                    use_seasonal_anchor=args.use_seasonal_anchor,
                    seasonal_half_life=args.seasonal_half_life,
                    seasonal_ridge=args.seasonal_ridge,
                    seasonal_harmonic_order=args.seasonal_harmonic_order,
                    annual_period_days=args.annual_period_days,
                    validation_blocks=args.validation_blocks,
                    worst_block_weight=args.worst_block_weight,
                    holdout_fraction=args.holdout_fraction,
                    min_holdout=args.min_holdout,
                    ridge=args.ridge,
                    mixture_ridge=args.mixture_ridge,
                    confidence_z=args.confidence_z,
                    risk_floor=args.risk_floor,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    max_blend=args.max_blend,
                    coefficient_fraction=args.coefficient_fraction,
                    strength=args.strength,
                    taper_power=args.taper_power,
                    protection_tolerance=args.protection_tolerance,
                ),
            )
        elif args.method == "sphere_ttc_v6":
            sphere = SphereTTCv6Calibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCv6Config(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    n_bands=args.n_bands,
                    memory_size=memory_size,
                    min_memory=args.min_memory,
                    short_memory_size=args.short_memory_size,
                    medium_memory_size=args.medium_memory_size,
                    short_half_life=args.short_half_life,
                    medium_half_life=args.medium_half_life,
                    long_half_life=args.long_half_life,
                    bias_anchor_half_life=args.bias_anchor_half_life,
                    use_bias_anchor=args.use_bias_anchor,
                    validation_blocks=args.validation_blocks,
                    worst_block_weight=args.worst_block_weight,
                    holdout_fraction=args.holdout_fraction,
                    min_holdout=args.min_holdout,
                    ridge=args.ridge,
                    mixture_ridge=args.mixture_ridge,
                    confidence_z=args.confidence_z,
                    risk_floor=args.risk_floor,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    max_blend=args.max_blend,
                    coefficient_fraction=args.coefficient_fraction,
                    strength=args.strength,
                    taper_power=args.taper_power,
                    protection_tolerance=args.protection_tolerance,
                ),
            )
        elif args.method == "sphere_ttc_v5":
            sphere = SphereTTCv5Calibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCv5Config(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    n_bands=args.n_bands,
                    memory_size=memory_size,
                    min_memory=args.min_memory,
                    short_memory_size=args.short_memory_size,
                    medium_memory_size=args.medium_memory_size,
                    short_half_life=args.short_half_life,
                    medium_half_life=args.medium_half_life,
                    long_half_life=args.long_half_life,
                    holdout_fraction=args.holdout_fraction,
                    min_holdout=args.min_holdout,
                    ridge=args.ridge,
                    mixture_ridge=args.mixture_ridge,
                    confidence_z=args.confidence_z,
                    risk_floor=args.risk_floor,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    max_blend=args.max_blend,
                    coefficient_fraction=args.coefficient_fraction,
                    strength=args.strength,
                    taper_power=args.taper_power,
                    protection_tolerance=args.protection_tolerance,
                ),
            )
        elif args.method == "sphere_ttc_v4":
            sphere = SphereTTCv4Calibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCv4Config(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    n_bands=args.n_bands,
                    memory_size=memory_size,
                    min_memory=args.min_memory,
                    short_memory_size=args.short_memory_size,
                    short_half_life=args.short_half_life,
                    long_half_life=args.long_half_life,
                    expert_mode=args.expert_mode,
                    holdout_fraction=args.holdout_fraction,
                    min_holdout=args.min_holdout,
                    ridge=args.ridge,
                    mixture_ridge=args.mixture_ridge,
                    confidence_z=args.confidence_z,
                    risk_floor=args.risk_floor,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    max_blend=args.max_blend,
                    coefficient_fraction=args.coefficient_fraction,
                    strength=args.strength,
                    taper_power=args.taper_power,
                ),
            )
        elif args.method == "sphere_ttc_v3":
            sphere = SphereTTCv3Calibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCv3Config(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    n_bands=args.n_bands,
                    memory_size=memory_size,
                    min_memory=args.min_memory,
                    half_life=args.half_life,
                    holdout_fraction=args.holdout_fraction,
                    min_holdout=args.min_holdout,
                    ridge=args.ridge,
                    gate_ridge=args.gate_ridge,
                    confidence_z=args.confidence_z,
                    risk_floor=args.risk_floor,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    max_blend=args.max_blend,
                    coefficient_fraction=args.coefficient_fraction,
                    strength=args.strength,
                    taper_power=args.taper_power,
                ),
            )
        elif args.method == "sphere_ttc_v2":
            sphere = SphereTTCv2Calibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCv2Config(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    half_life=args.half_life,
                    ridge=args.ridge,
                    strength=args.strength,
                    min_memory=args.min_memory,
                    holdout_fraction=args.holdout_fraction,
                    validation_fraction=args.validation_fraction,
                    risk_fraction=args.risk_fraction,
                    min_holdout=args.min_holdout,
                    confidence_z=args.confidence_z,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    taper_power=args.taper_power,
                    n_bands=args.n_bands,
                    gate_ridge=args.gate_ridge,
                    max_component_blend=args.max_component_blend,
                    risk_floor=args.risk_floor,
                    hierarchy_weight=args.hierarchy_weight,
                ),
            )
        else:
            sphere = SphereTTCCalibrator(
                n_lat,
                n_lon,
                device,
                SphereTTCConfig(
                    lmax=args.lmax,
                    mmax=args.mmax,
                    half_life=args.half_life,
                    ridge=args.ridge,
                    strength=args.strength,
                    min_memory=args.min_memory,
                    holdout_fraction=args.holdout_fraction,
                    min_holdout=args.min_holdout,
                    confidence_z=args.confidence_z,
                    robust_quantile_scale=args.robust_quantile_scale,
                    max_scale_delta=args.max_scale_delta,
                    taper_power=args.taper_power,
                ),
            )
    elif args.method == "st_ttc":
        stats = json.loads(Path(args.normalization_stats).read_text(encoding="utf-8"))
        stats_variables = list(stats["variables"])
        stats_indices = [stats_variables.index(name) for name in selected_variables]
        st_mean = torch.as_tensor(
            np.asarray(stats["mean"], dtype=np.float32)[stats_indices],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, n_var, 1, 1)
        st_std = torch.as_tensor(
            np.asarray(stats["std"], dtype=np.float32)[stats_indices],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, n_var, 1, 1)
        st_ttc = STTTCSDCalibrator(
            n_lead,
            n_var * n_lat * n_lon,
            groups=args.st_ttc_groups,
        ).to(device)
        st_optimizer = torch.optim.Adam(st_ttc.parameters(), lr=float(args.st_ttc_lr))
        weights = latitude_weights(ds["lat"].values, device)

    def update_st_ttc(current):
        if args.method != "st_ttc":
            return 0
        eligible = [item for item in st_buffer if np.datetime64(item[0]) <= np.datetime64(current)]
        eligible = eligible[-int(memory_size) :]
        if not eligible:
            return 0
        available_time, pred_old, target_old = eligible[-1]
        if st_last_update[0] is not None and np.datetime64(available_time) <= np.datetime64(st_last_update[0]):
            return len(eligible)
        assert_no_future_targets(current, [available_time])
        pred_normalized = (pred_old.unsqueeze(0) - st_mean) / st_std
        target_normalized = (target_old.unsqueeze(0) - st_mean) / st_std
        st_ttc.train()
        for _ in range(int(args.st_ttc_updates)):
            st_optimizer.zero_grad(set_to_none=True)
            calibrated_old = st_ttc(pred_normalized)
            loss = weighted_mean_spatial(
                (calibrated_old - target_normalized).square(),
                weights,
            ).mean()
            loss.backward()
            st_optimizer.step()
        st_last_update[0] = available_time
        return len(eligible)

    if args.warmup_cache:
        if args.trim_warmup_cache and args.method not in {
            "sphere_ttc",
            "sphere_ttc_v2",
            "sphere_ttc_v3",
            "sphere_ttc_v4",
            "sphere_ttc_v5",
            "sphere_ttc_v6",
            "sphere_ttc_v7",
        }:
            raise RuntimeError("--trim-warmup-cache is currently validated only for sphere_ttc")
        warmup_datasets = []
        previous_max = None
        for warmup_path in args.warmup_cache:
            warm = xr.open_zarr(warmup_path, consolidated=True)
            warm = _restrict_warmup_period(
                warm, args.warmup_end_exclusive
            )
            if args.variables:
                warm = warm.sel(variable=args.variables)
            if args.lead_time is not None:
                warm = warm.sel(lead_time=[args.lead_time])
            for coord in ("lead_time", "variable", "lat", "lon"):
                if not np.array_equal(np.asarray(warm[coord].values), np.asarray(ds[coord].values)):
                    raise RuntimeError(f"warmup cache {coord} does not match test cache")
            warm_times = np.asarray(warm["init_time"].values)
            if warm_times.max() >= np.asarray(init_times).min():
                raise RuntimeError("warmup cache must end before the first test initialization")
            if previous_max is not None and warm_times.min() <= previous_max:
                raise RuntimeError("warmup caches must be strictly chronological and non-overlapping")
            previous_max = warm_times.max()
            warmup_datasets.append((warmup_path, warm, warm_times))

        warmup_starts = [0] * len(warmup_datasets)
        if args.trim_warmup_cache:
            remaining = keep_size
            for source_index in range(len(warmup_datasets) - 1, -1, -1):
                source_size = warmup_datasets[source_index][1].sizes["init_time"]
                take = min(source_size, remaining)
                warmup_starts[source_index] = source_size - take
                remaining -= take
                if remaining == 0:
                    for earlier in range(source_index):
                        warmup_starts[earlier] = warmup_datasets[earlier][1].sizes["init_time"]
                    break

        for (warmup_path, warm, warm_times), warmup_start in zip(
            warmup_datasets, warmup_starts
        ):
            warm_valid_times = np.asarray(warm["valid_time"].values)
            for wi in range(warmup_start, warm.sizes["init_time"]):
                warm_current = warm_times[wi]
                pred_np = np.asarray(warm["pred"].isel(init_time=wi).values, dtype=np.float32)
                warmup_truth_root = (
                    args.warmup_truth_root or args.truth_root
                )
                if warmup_truth_root:
                    from src.data_sources.s2s_daily import load_targets

                    target_np = load_targets(
                        warmup_truth_root,
                        [warm_current],
                        lead_values,
                        selected_variables,
                        workers=args.truth_workers,
                    )[0]
                else:
                    target_np = np.asarray(warm["target"].isel(init_time=wi).values, dtype=np.float32)
                pred_t = _as_torch(pred_np, device)
                target_t = _as_torch(target_np, device)
                if args.method == "st_ttc":
                    update_st_ttc(warm_current)
                    st_buffer.append(
                        (
                            np.asarray(warm_valid_times[wi]).max(),
                            pred_t.detach(),
                            target_t.detach(),
                        )
                    )
                    if len(st_buffer) > keep_size:
                        st_buffer[:] = st_buffer[-keep_size:]
                elif args.method in {
                    "sphere_ttc",
                    "sphere_ttc_v2",
                    "sphere_ttc_v3",
                    "sphere_ttc_v4",
                    "sphere_ttc_v5",
                    "sphere_ttc_v6",
                    "sphere_ttc_v7",
                }:
                    pred_coeff = sphere.encode(pred_t)
                    target_coeff = sphere.encode(target_t)
                    for lead_idx in range(n_lead):
                        coefficient_buffers[lead_idx].append(
                            (
                                warm_valid_times[wi, lead_idx],
                                pred_coeff[lead_idx].detach(),
                                target_coeff[lead_idx].detach(),
                            )
                        )
                        if len(coefficient_buffers[lead_idx]) > keep_size:
                            coefficient_buffers[lead_idx] = coefficient_buffers[lead_idx][-keep_size:]
                elif args.method in {"static_bias", "ema_bias"}:
                    for lead_idx in range(n_lead):
                        residual_buffers[lead_idx].append(
                            (
                                warm_valid_times[wi, lead_idx],
                                _finite_residual(target_t[lead_idx], pred_t[lead_idx]).detach(),
                            )
                        )
                        if len(residual_buffers[lead_idx]) > keep_size:
                            residual_buffers[lead_idx] = residual_buffers[lead_idx][-keep_size:]
                else:
                    for lead_idx in range(n_lead):
                        pred_mem, target_mem = _finite_memory_pair(pred_t[lead_idx], target_t[lead_idx])
                        sample_buffers[lead_idx].append(
                            (warm_valid_times[wi, lead_idx], pred_mem.detach(), target_mem.detach())
                        )
                        if len(sample_buffers[lead_idx]) > keep_size:
                            sample_buffers[lead_idx] = sample_buffers[lead_idx][-keep_size:]
                del pred_t, target_t
            print(
                f"{args.model} loaded causal warmup cache {warmup_path} "
                f"with {warm.sizes['init_time'] - warmup_start}/"
                f"{warm.sizes['init_time']} retained initializations",
                flush=True,
            )

    counts = np.zeros((n_init, n_lead), dtype=np.int32)
    gates = np.zeros((n_init, n_lead, n_var), dtype=np.float32)
    sphere_lead_executor = (
        ThreadPoolExecutor(max_workers=min(n_lead, max(1, args.cpu_lead_workers)))
        if args.method in {
            "sphere_ttc",
            "sphere_ttc_v2",
            "sphere_ttc_v3",
            "sphere_ttc_v4",
            "sphere_ttc_v5",
            "sphere_ttc_v6",
            "sphere_ttc_v7",
        }
        and device.type == "cpu"
        and args.cpu_lead_workers > 1
        else None
    )

    def calibrate_sphere_lead(
        lead_idx,
        current,
        current_valid,
        pred_current,
        current_coeff,
    ):
        eligible = _eligible_from_buffer(
            coefficient_buffers[lead_idx], current, memory_size
        )
        if not eligible:
            return pred_current, None, 0
        assert_no_future_targets(current, [item[0] for item in eligible])
        pred_mem_coeff = torch.stack([item[1] for item in eligible], dim=0)
        target_mem_coeff = torch.stack([item[2] for item in eligible], dim=0)
        if args.method == "sphere_ttc_v7":
            memory_days = torch.as_tensor(
                [
                    int(np.datetime64(item[0], "D").astype(np.int64))
                    for item in eligible
                ],
                dtype=torch.float32,
                device=device,
            )
            current_day = int(
                np.datetime64(current_valid, "D").astype(np.int64)
            )
            calibrated_coeff, current_gate = sphere.calibrate_coefficients(
                current_coeff,
                pred_mem_coeff,
                target_mem_coeff,
                memory_days,
                current_day,
            )
        else:
            calibrated_coeff, current_gate = sphere.calibrate_coefficients(
                current_coeff,
                pred_mem_coeff,
                target_mem_coeff,
            )
        correction = sphere.isht(calibrated_coeff - current_coeff)
        current_safe = torch.nan_to_num(pred_current.float())
        calibrated = torch.where(
            torch.isfinite(pred_current),
            current_safe + correction,
            pred_current,
        )
        return calibrated, current_gate, len(eligible)

    for i, current in enumerate(init_times):
        pred_i_np = np.asarray(ds["pred"].isel(init_time=i).values, dtype=np.float32)
        if args.truth_root:
            from src.data_sources.s2s_daily import load_targets

            target_i_np = load_targets(
                args.truth_root,
                [current],
                lead_values,
                selected_variables,
                workers=args.truth_workers,
            )[0]
        else:
            target_i_np = np.asarray(ds["target"].isel(init_time=i).values, dtype=np.float32)
        pred_i_t = _as_torch(pred_i_np, device)
        target_i_t = _as_torch(target_i_np, device)
        st_calibrated_i = None
        pred_coeff_i = target_coeff_i = None
        if args.method in {
            "sphere_ttc",
            "sphere_ttc_v2",
            "sphere_ttc_v3",
            "sphere_ttc_v4",
            "sphere_ttc_v5",
            "sphere_ttc_v6",
            "sphere_ttc_v7",
        }:
            # Batch the five lead transforms once. This is mathematically
            # identical to encoding each lead inside calibrate(), and avoids
            # encoding the current prediction a second time for memory.
            pred_coeff_i = sphere.encode(pred_i_t)
            target_coeff_i = sphere.encode(target_i_t)
        if args.method == "st_ttc":
            eligible_count = update_st_ttc(current)
            counts[i, :] = eligible_count
            st_ttc.eval()
            with torch.no_grad():
                normalized = (pred_i_t.unsqueeze(0) - st_mean) / st_std
                st_calibrated_i = st_ttc(normalized)[0] * st_std[0] + st_mean[0]
        sphere_lead_results = None
        if sphere_lead_executor is not None:
            futures = [
                sphere_lead_executor.submit(
                    calibrate_sphere_lead,
                    lead_idx,
                    current,
                    valid_times[i, lead_idx],
                    pred_i_t[lead_idx],
                    pred_coeff_i[lead_idx],
                )
                for lead_idx in range(n_lead)
            ]
            sphere_lead_results = [future.result() for future in futures]
        if out_group is not None and not args.omit_target:
            out_group["target"][i, :, :, :, :] = target_i_np
        for lead_idx in range(n_lead):
            pred_current = pred_i_t[lead_idx]
            if args.method == "raw":
                calibrated = pred_current
            elif args.method == "st_ttc":
                calibrated = st_calibrated_i[lead_idx]
            elif args.method == "static_bias":
                eligible = _eligible_from_buffer(residual_buffers[lead_idx], current, memory_size)
                counts[i, lead_idx] = len(eligible)
                if eligible:
                    assert_no_future_targets(current, [item[0] for item in eligible])
                    residual = torch.stack([item[1] for item in eligible], dim=0).mean(dim=0)
                    calibrated = pred_current + residual
                else:
                    calibrated = pred_current
            elif args.method == "ema_bias":
                eligible = _eligible_from_buffer(residual_buffers[lead_idx], current, memory_size)
                counts[i, lead_idx] = len(eligible)
                if eligible:
                    assert_no_future_targets(current, [item[0] for item in eligible])
                    residual = torch.stack([item[1] for item in eligible], dim=0).mean(dim=0)
                    ema_state[lead_idx] = (1.0 - float(cfg["ttc"]["alpha"])) * ema_state[lead_idx] + float(cfg["ttc"]["alpha"]) * residual
                calibrated = pred_current + ema_state[lead_idx]
            elif args.method == "affine_ttc":
                eligible = _eligible_from_buffer(sample_buffers[lead_idx], current, memory_size)
                counts[i, lead_idx] = len(eligible)
                if eligible:
                    assert_no_future_targets(current, [item[0] for item in eligible])
                    pred_mem = torch.stack([item[1] for item in eligible], dim=0)
                    target_mem = torch.stack([item[2] for item in eligible], dim=0)
                    for _ in range(int(adapt_steps)):
                        opt.zero_grad(set_to_none=True)
                        yhat = pred_mem * (1.0 + scale[lead_idx]) + bias[lead_idx]
                        loss = weighted_mean_spatial((yhat - target_mem) ** 2, weights).mean()
                        loss = loss + float(cfg["ttc"]["lambda_scale"]) * (scale[lead_idx] ** 2).mean()
                        loss = loss + float(cfg["ttc"]["lambda_bias"]) * (bias[lead_idx] ** 2).mean()
                        loss.backward()
                        opt.step()
                calibrated = pred_current * (1.0 + scale[lead_idx].detach()) + bias[lead_idx].detach()
            elif args.method == "zonal_spectral_ttc":
                eligible = _eligible_from_buffer(sample_buffers[lead_idx], current, memory_size)
                counts[i, lead_idx] = len(eligible)
                if eligible:
                    assert_no_future_targets(current, [item[0] for item in eligible])
                    pred_mem = torch.stack([item[1] for item in eligible], dim=0)
                    target_mem = torch.stack([item[2] for item in eligible], dim=0)
                    for _ in range(int(adapt_steps)):
                        opt.zero_grad(set_to_none=True)
                        yhat = apply_zonal_spectral(pred_mem.float(), gate[lead_idx], bias[lead_idx], k)
                        loss = weighted_mean_spatial((yhat - target_mem.float()) ** 2, weights).mean()
                        loss.backward()
                        opt.step()
                calibrated = apply_zonal_spectral(pred_current[None].float(), gate[lead_idx].detach(), bias[lead_idx].detach(), k)[0]
            elif args.method in {
                "sphere_ttc",
                "sphere_ttc_v2",
                "sphere_ttc_v3",
                "sphere_ttc_v4",
                "sphere_ttc_v5",
                "sphere_ttc_v6",
                "sphere_ttc_v7",
            }:
                if sphere_lead_results is None:
                    calibrated, current_gate, eligible_count = calibrate_sphere_lead(
                        lead_idx,
                        current,
                        valid_times[i, lead_idx],
                        pred_current,
                        pred_coeff_i[lead_idx],
                    )
                else:
                    calibrated, current_gate, eligible_count = sphere_lead_results[
                        lead_idx
                    ]
                counts[i, lead_idx] = eligible_count
                if current_gate is not None:
                    gates[i, lead_idx] = current_gate.detach().cpu().numpy()
            else:
                eligible = _eligible_from_buffer(sample_buffers[lead_idx], current, memory_size)
                counts[i, lead_idx] = len(eligible)
                if eligible:
                    assert_no_future_targets(current, [item[0] for item in eligible])
                    pred_mem = torch.stack([item[1] for item in eligible], dim=0)
                    target_mem = torch.stack([item[2] for item in eligible], dim=0)
                    calibrated = apply_geo_ttc(
                        pred_current,
                        pred_mem,
                        target_mem,
                        n_modes=k,
                        snr_lambda=args.snr_lambda,
                        strength=args.strength,
                        min_memory=args.min_memory,
                        sample_tau=args.sample_tau,
                    )
                else:
                    calibrated = pred_current
            if args.metrics_output:
                difference = calibrated - target_i_t[lead_idx]
                metric_mse[i, lead_idx] = (
                    weighted_mean_spatial(difference.square(), metric_weights).detach().cpu().numpy()
                )
                metric_mae[i, lead_idx] = (
                    weighted_mean_spatial(difference.abs(), metric_weights).detach().cpu().numpy()
                )
                metric_bias[i, lead_idx] = (
                    weighted_mean_spatial(difference, metric_weights).detach().cpu().numpy()
                )
                if (
                    metric_acc_climatology is not None
                    or metric_acc_climatology_by_day is not None
                ):
                    if metric_acc_climatology_by_day is not None:
                        current_climatology = _as_torch(
                            metric_acc_climatology_by_day[
                                metric_acc_day_indices[i, lead_idx]
                            ],
                            device,
                        )
                    else:
                        current_climatology = metric_acc_climatology[lead_idx]
                    prediction_anomaly = calibrated - current_climatology
                    target_anomaly = (
                        target_i_t[lead_idx] - current_climatology
                    )
                    metric_acc_numerator[i, lead_idx] = (
                        weighted_mean_spatial(
                            prediction_anomaly * target_anomaly,
                            metric_weights,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    metric_acc_prediction_energy[i, lead_idx] = (
                        weighted_mean_spatial(
                            prediction_anomaly.square(),
                            metric_weights,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    metric_acc_target_energy[i, lead_idx] = (
                        weighted_mean_spatial(
                            target_anomaly.square(),
                            metric_weights,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
            if out_group is not None:
                out_group["pred"][i, lead_idx, :, :, :] = calibrated.detach().cpu().numpy().astype("float32")
            del pred_current, calibrated
        if out_group is not None:
            out_group["eligible_memory_count"][i, :] = counts[i]
            out_group["calibration_gate"][i, :, :] = gates[i]
        if args.method == "st_ttc":
            st_buffer.append(
                (
                    np.asarray(valid_times[i]).max(),
                    pred_i_t.detach(),
                    target_i_t.detach(),
                )
            )
            if len(st_buffer) > keep_size:
                st_buffer[:] = st_buffer[-keep_size:]
        for lead_idx in range(n_lead):
            valid = valid_times[i, lead_idx]
            if args.method in {"static_bias", "ema_bias"}:
                residual = _finite_residual(target_i_t[lead_idx], pred_i_t[lead_idx])
                residual_buffers[lead_idx].append((valid, residual.detach()))
                if len(residual_buffers[lead_idx]) > keep_size:
                    residual_buffers[lead_idx] = residual_buffers[lead_idx][-keep_size:]
            elif args.method in {
                "sphere_ttc",
                "sphere_ttc_v2",
                "sphere_ttc_v3",
                "sphere_ttc_v4",
                "sphere_ttc_v5",
                "sphere_ttc_v6",
                "sphere_ttc_v7",
            }:
                coefficient_buffers[lead_idx].append(
                    (
                        valid,
                        pred_coeff_i[lead_idx].detach(),
                        target_coeff_i[lead_idx].detach(),
                    )
                )
                if len(coefficient_buffers[lead_idx]) > keep_size:
                    coefficient_buffers[lead_idx] = coefficient_buffers[lead_idx][-keep_size:]
            elif args.method == "st_ttc":
                pass
            elif args.method != "raw":
                pred_mem, target_mem = _finite_memory_pair(pred_i_t[lead_idx], target_i_t[lead_idx])
                sample_buffers[lead_idx].append((valid, pred_mem.detach(), target_mem.detach()))
                if len(sample_buffers[lead_idx]) > keep_size:
                    sample_buffers[lead_idx] = sample_buffers[lead_idx][-keep_size:]
        del pred_i_t, target_i_t
        if (i + 1) % 16 == 0 or i + 1 == n_init:
            print(f"{args.model} {args.method} wrote init {i + 1}/{n_init}", flush=True)

    if out_group is not None:
        zarr.consolidate_metadata(str(out))
    if sphere_lead_executor is not None:
        sphere_lead_executor.shutdown()
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "init_time": np.asarray(init_times).astype("datetime64[ns]").astype(np.int64),
            "lead_time": np.asarray(lead_values, dtype=np.int32),
            "variable": np.asarray(selected_variables),
            "mse": metric_mse,
            "mae": metric_mae,
            "bias": metric_bias,
            "eligible_memory_count": counts,
            "calibration_gate": gates,
        }
        if (
            metric_acc_climatology is not None
            or metric_acc_climatology_by_day is not None
        ):
            metrics_payload.update(
                {
                    "acc_numerator": metric_acc_numerator,
                    "acc_prediction_energy": metric_acc_prediction_energy,
                    "acc_target_energy": metric_acc_target_energy,
                }
            )
        np.savez_compressed(metrics_path, **metrics_payload)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    profile = {
        "stage": "ttc",
        "model_name": args.model,
        "method": args.method,
        "input_cache": str(cache),
        "output_cache": None if args.no_cache_output else str(out),
        "metrics_output": args.metrics_output,
        "output_includes_target": not args.omit_target,
        "variables": args.variables,
        "truth_root": args.truth_root,
        "init_times_from": args.init_times_from,
        "init_start": args.init_start,
        "init_end_inclusive": args.init_end_inclusive,
        "warmup_cache": args.warmup_cache,
        "memory_size": memory_size,
        "n_modes": None if k is None else int(k),
        "snr_lambda": args.snr_lambda if args.method == "geo_ttc" else None,
        "strength": args.strength if args.method == "geo_ttc" else None,
        "min_memory": args.min_memory if args.method == "geo_ttc" else None,
        "sample_tau": args.sample_tau if args.method == "geo_ttc" else None,
        "sphere_ttc": (
            {
                "lmax": args.lmax,
                "mmax": args.mmax,
                "half_life": args.half_life,
                "ridge": args.ridge,
                "strength": args.strength,
                "min_memory": args.min_memory,
                "holdout_fraction": args.holdout_fraction,
                "min_holdout": args.min_holdout,
                "confidence_z": args.confidence_z,
                "robust_quantile_scale": args.robust_quantile_scale,
                "max_scale_delta": args.max_scale_delta,
                "taper_power": args.taper_power,
                "validation_fraction": (
                    args.validation_fraction
                    if args.method == "sphere_ttc_v2"
                    else None
                ),
                "risk_fraction": (
                    args.risk_fraction if args.method == "sphere_ttc_v2" else None
                ),
                "n_bands": (
                    args.n_bands
                    if args.method in {
                        "sphere_ttc_v2",
                        "sphere_ttc_v3",
                        "sphere_ttc_v4",
                        "sphere_ttc_v5",
                        "sphere_ttc_v6",
                        "sphere_ttc_v7",
                        "sphere_ttc_v7",
                    }
                    else None
                ),
                "gate_ridge": (
                    args.gate_ridge
                    if args.method in {
                        "sphere_ttc_v2",
                        "sphere_ttc_v3",
                        "sphere_ttc_v4",
                        "sphere_ttc_v5",
                        "sphere_ttc_v6",
                        "sphere_ttc_v7",
                    }
                    else None
                ),
                "max_component_blend": (
                    args.max_component_blend
                    if args.method == "sphere_ttc_v2"
                    else None
                ),
                "risk_floor": (
                    args.risk_floor
                    if args.method in {
                        "sphere_ttc_v2",
                        "sphere_ttc_v3",
                        "sphere_ttc_v4",
                        "sphere_ttc_v5",
                        "sphere_ttc_v6",
                        "sphere_ttc_v7",
                    }
                    else None
                ),
                "hierarchy_weight": (
                    args.hierarchy_weight
                    if args.method == "sphere_ttc_v2"
                    else None
                ),
                "max_blend": (
                    args.max_blend
                    if args.method in {
                        "sphere_ttc_v3",
                        "sphere_ttc_v4",
                        "sphere_ttc_v5",
                        "sphere_ttc_v6",
                        "sphere_ttc_v7",
                    }
                    else None
                ),
                "coefficient_fraction": (
                    args.coefficient_fraction
                    if args.method in {
                        "sphere_ttc_v3",
                        "sphere_ttc_v4",
                        "sphere_ttc_v5",
                        "sphere_ttc_v6",
                        "sphere_ttc_v7",
                    }
                    else None
                ),
                "short_memory_size": (
                    args.short_memory_size
                    if args.method in {"sphere_ttc_v4", "sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "medium_memory_size": (
                    args.medium_memory_size
                    if args.method in {"sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "short_half_life": (
                    args.short_half_life
                    if args.method in {"sphere_ttc_v4", "sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "medium_half_life": (
                    args.medium_half_life
                    if args.method in {"sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "long_half_life": (
                    args.long_half_life
                    if args.method in {"sphere_ttc_v4", "sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "mixture_ridge": (
                    args.mixture_ridge
                    if args.method in {"sphere_ttc_v4", "sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "protection_tolerance": (
                    args.protection_tolerance
                    if args.method in {"sphere_ttc_v5", "sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "bias_anchor_half_life": (
                    args.bias_anchor_half_life
                    if args.method == "sphere_ttc_v6"
                    else None
                ),
                "use_bias_anchor": (
                    args.use_bias_anchor
                    if args.method == "sphere_ttc_v6"
                    else None
                ),
                "validation_blocks": (
                    args.validation_blocks
                    if args.method in {"sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "worst_block_weight": (
                    args.worst_block_weight
                    if args.method in {"sphere_ttc_v6", "sphere_ttc_v7"}
                    else None
                ),
                "use_seasonal_anchor": (
                    args.use_seasonal_anchor
                    if args.method == "sphere_ttc_v7"
                    else None
                ),
                "seasonal_half_life": (
                    args.seasonal_half_life
                    if args.method == "sphere_ttc_v7"
                    else None
                ),
                "seasonal_ridge": (
                    args.seasonal_ridge
                    if args.method == "sphere_ttc_v7"
                    else None
                ),
                "seasonal_harmonic_order": (
                    args.seasonal_harmonic_order
                    if args.method == "sphere_ttc_v7"
                    else None
                ),
                "annual_period_days": (
                    args.annual_period_days
                    if args.method == "sphere_ttc_v7"
                    else None
                ),
                "expert_mode": (
                    args.expert_mode
                    if args.method == "sphere_ttc_v4"
                    else None
                ),
            }
            if args.method in {
                "sphere_ttc",
                "sphere_ttc_v2",
                "sphere_ttc_v3",
                "sphere_ttc_v4",
                "sphere_ttc_v5",
                "sphere_ttc_v6",
                "sphere_ttc_v7",
            }
            else None
        ),
        "st_ttc": (
            {
                "groups": args.st_ttc_groups,
                "lr": args.st_ttc_lr,
                "updates": args.st_ttc_updates,
                "normalization_stats": args.normalization_stats,
                "full_horizon_feedback_delay": True,
            }
            if args.method == "st_ttc"
            else None
        ),
        "pred_shape": list(shape),
        "wall_seconds": elapsed,
        "cuda_peak_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0.0
        ),
        "device_info": get_device_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    profile_name = f"{args.model}_{args.method}.json"
    if args.output:
        output_path = Path(args.output)
        profile_name = f"{output_path.parent.name}_{output_path.stem}.json"
    profile_path = (
        Path(args.profile_output)
        if args.profile_output
        else Path("artifacts/profiles/ttc") / profile_name
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"WROTE {metrics_path if args.no_cache_output else out}")
    print(f"PROFILE {profile_path} wall_seconds={elapsed:.3f} cuda_peak_memory_gb={profile['cuda_peak_memory_gb']:.3f}")


if __name__ == "__main__":
    main()
