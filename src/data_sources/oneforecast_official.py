from __future__ import annotations

import sys
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.daily_mean import six_hour_step_to_lead_indices
from src.utils.device import get_device_info
from src.utils.time import make_valid_time
from src.utils.xarray_utils import normalize_lat_lon, open_zarr_lazy, select_canonical_dataarray


ONEFORECAST_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
ONEFORECAST_CHANNELS = {
    **{f"z{level}": i for i, level in enumerate(ONEFORECAST_LEVELS)},
    **{f"q{level}": 13 + i for i, level in enumerate(ONEFORECAST_LEVELS)},
    **{f"t{level}": 26 + i for i, level in enumerate(ONEFORECAST_LEVELS)},
    **{f"u{level}": 39 + i for i, level in enumerate(ONEFORECAST_LEVELS)},
    **{f"v{level}": 52 + i for i, level in enumerate(ONEFORECAST_LEVELS)},
    "u10": 65,
    "v10": 66,
    "t2m": 67,
    "mslp": 68,
}
# The official checkpoint has 69 channels: five atmospheric fields at all 13
# pressure levels and four surface fields. It does not predict precipitation.
ONEFORECAST_EVAL_CHANNELS = dict(ONEFORECAST_CHANNELS)
ONEFORECAST_LONGITUDE_ROLL = 120


def inspect_official(root: str = "external/OneForecast"):
    missing = []
    root_path = Path(root)
    if not root_path.exists():
        missing.append("missing_official_repo")
    required = [
        "data/global/mean.npy",
        "data/global/std.npy",
        "exp/OneForecast/20241008-171138/config.yaml",
        "exp/OneForecast/20241008-171138/training_checkpoints/best_ckpt.tar",
    ]
    for rel in required:
        if not (root_path / rel).exists():
            missing.append(f"missing_{rel}")
    try:
        import dgl  # noqa: F401
        import h5py  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as exc:
        missing.append(f"missing_dependency:{type(exc).__name__}:{exc}")
    return {
        "status": "unavailable" if missing else "available_official_inference",
        "source_uri": "https://github.com/YuanGao-YG/OneForecast",
        "available_variables": list(ONEFORECAST_EVAL_CHANNELS),
        "available_leads": [6 * i for i in range(1, 41)],
        "available_times": {"count": None, "start": None, "end": None},
        "missing_requirements": missing,
        "notes": "Uses the official checkpoint/statistics with canonical ERA5 initial states and canonical ERA5 targets.",
    }


def _load_model(root: Path, device):
    import torch

    sys.path.insert(0, str(root.resolve()))
    from my_utils.YParams import YParams
    from models.OneForecast import OneForecast

    config = root / "exp/OneForecast/20241008-171138/config.yaml"
    params = YParams(str(config), "OneForecast")
    params["img_shape_x"] = 120
    params["img_shape_y"] = 240
    params["N_in_channels"] = len(params.in_channels)
    params["N_out_channels"] = len(params.out_channels)
    model = OneForecast(params).to(device).eval()
    ckpt = torch.load(
        root / "exp/OneForecast/20241008-171138/training_checkpoints/best_ckpt.tar",
        map_location=device,
        weights_only=False,
    )
    state = {}
    for key, val in ckpt["model_state"].items():
        name = key[7:] if key.startswith("module.") else key
        if name != "ged":
            state[name] = val
    model.load_state_dict(state)
    return model, params


def _canonical_era5_targets(truth, variables: list[str], init_times: np.ndarray, lead_hours: list[int]):
    """Return canonical WeatherBench2 ERA5 targets for the requested forecast times."""
    import xarray as xr

    valid = make_valid_time(init_times, lead_hours)
    by_variable = []
    for variable in variables:
        base = select_canonical_dataarray(truth, variable)
        by_lead = []
        for lead_index, lead in enumerate(lead_hours):
            values = base.sel(time=valid[:, lead_index]).rename({"time": "init_time"})
            values = values.assign_coords(init_time=init_times, lead_time=np.int32(lead))
            by_lead.append(values)
        stacked = xr.concat(by_lead, dim="lead_time")
        by_variable.append(stacked)
    target = xr.concat(by_variable, dim="variable", coords="minimal", compat="override").assign_coords(variable=variables)
    return target.transpose("init_time", "lead_time", "variable", "lat", "lon").astype("float32")


def _canonical_era5_initial_state(truth, init_time: np.datetime64):
    """Build one official-model input state from ERA5 at the init time only."""
    arrays = []
    for variable in ONEFORECAST_CHANNELS:
        values = select_canonical_dataarray(truth, variable).sel(time=init_time)
        if set(("lat", "lon")).issubset(values.dims):
            values = values.transpose("lat", "lon")
        arrays.append(np.asarray(values.values, dtype=np.float32))
    return np.stack(arrays, axis=0).astype(np.float32, copy=False)


def _to_oneforecast_longitude_order(values: np.ndarray) -> np.ndarray:
    """Convert 0..358.5 WeatherBench order to the checkpoint's -180..178.5 order."""
    return np.roll(values, ONEFORECAST_LONGITUDE_ROLL, axis=-1)


def _from_oneforecast_longitude_order(values: np.ndarray) -> np.ndarray:
    """Convert checkpoint-native -180..178.5 order back to 0..358.5."""
    return np.roll(values, -ONEFORECAST_LONGITUDE_ROLL, axis=-1)


def build_oneforecast_cache(cfg: dict, output_path: str):
    import torch
    import xarray as xr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OneForecast official inference.")
    start_total = time.perf_counter()
    device = torch.device("cuda")
    root = Path("external/OneForecast")
    model, params = _load_model(root, device)
    requested_variables = [v for v in cfg["experiment"]["variables"] if v in ONEFORECAST_EVAL_CHANNELS]
    if not requested_variables:
        raise RuntimeError("No requested variables are available in OneForecast channel map.")
    channels = np.asarray([ONEFORECAST_EVAL_CHANNELS[v] for v in requested_variables], dtype=np.int64)
    lead_hours = [int(x) for x in cfg["experiment"]["lead_time_hours"]]
    step_to_leads = six_hour_step_to_lead_indices(lead_hours)
    max_step = max(step_to_leads)
    year = int(cfg["experiment"]["year"])
    init_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00")
    frequency = int(cfg["experiment"].get("init_frequency_hours", 24))
    init_times = pd.date_range(f"{year}-01-01T00:00:00", init_end, freq=f"{frequency}h", inclusive="left").values.astype("datetime64[ns]")
    max_init_times = cfg["experiment"].get("max_init_times")
    if max_init_times is not None:
        init_times = init_times[: int(max_init_times)]
    mean = np.load(root / "data/global/mean.npy").astype("float32")
    std = np.load(root / "data/global/std.npy").astype("float32")
    native_lat = 120
    rollout_seconds = 0.0
    io_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    native_lat_coord = np.linspace(90.0, -90.0, 121, dtype=np.float32)[:native_lat]
    target_lat = np.linspace(90.0, -90.0, 121, dtype=np.float32)
    lon = np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32)
    truth = open_zarr_lazy(cfg["weatherbench2"]["era5_truth_uri"])
    truth, _ = normalize_lat_lon(truth)
    out_path = Path(output_path)
    resume = bool(cfg["experiment"].get("official_inference_resume", False))
    completed_init_times = 0
    if out_path.exists() and resume:
        existing = xr.open_zarr(out_path, consolidated=False)
        completed_init_times = int(existing.sizes.get("init_time", 0))
        if completed_init_times > len(init_times):
            raise RuntimeError(
                f"OneForecast resume cache has {completed_init_times} initializations, "
                f"expected at most {len(init_times)}"
            )
        existing_times = np.asarray(existing["init_time"].values)
        expected_times = init_times[:completed_init_times]
        if not np.array_equal(existing_times, expected_times):
            raise RuntimeError("OneForecast resume cache init_time prefix does not match the protocol")
        existing.close()
        print(
            f"OneForecast resuming at {completed_init_times}/{len(init_times)} initializations",
            flush=True,
        )
    elif out_path.exists():
        shutil.rmtree(out_path)
    batch_size = int(cfg["experiment"].get("official_inference_batch_size", 8))
    # The released checkpoint is numerically unstable under FP16 autocast on
    # the V100 environment used by this experiment: rollouts can silently
    # become entirely NaN while the process still exits successfully. Keep
    # official inference in FP32 unless a future protocol explicitly opts in.
    use_amp = bool(cfg["experiment"].get("official_inference_amp", False))
    first_write = completed_init_times == 0
    with torch.no_grad():
        for batch_start in range(completed_init_times, len(init_times), batch_size):
                batch_end = min(batch_start + batch_size, len(init_times))
                batch_init_times = init_times[batch_start:batch_end]
                pred = np.zeros((len(batch_init_times), len(lead_hours), len(channels), native_lat, 240), dtype=np.float32)
                print(f"OneForecast batch {batch_start + 1}-{batch_end}/{len(init_times)}", flush=True)
                for bi, init_time in enumerate(batch_init_times):
                    global_i = batch_start + bi
                    print(f"OneForecast init {global_i + 1}/{len(init_times)} time={pd.to_datetime(init_time)}: loading ERA5 input", flush=True)
                    t0 = time.perf_counter()
                    current = _canonical_era5_initial_state(truth, init_time)
                    current = _to_oneforecast_longitude_order(current)[:, :native_lat, :]
                    io_seconds += time.perf_counter() - t0
                    norm = (current[None] - mean) / std
                    x = torch.as_tensor(norm, dtype=torch.float32, device=device)
                    print(f"OneForecast init {global_i + 1}/{len(init_times)}: rollout {max_step} steps", flush=True)
                    t1 = time.perf_counter()
                    for step in range(1, max_step + 1):
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                # OneForecast's reference forward uses view()
                                # internally; AMP graph outputs can be
                                # non-contiguous after the first step.
                                x = model(x.contiguous())
                        else:
                            x = model(x.contiguous())
                        if step in step_to_leads:
                            denorm = x.float().detach().cpu().numpy() * std + mean
                            if not np.isfinite(denorm).all():
                                raise FloatingPointError(
                                    "OneForecast produced non-finite values "
                                    f"at init={pd.to_datetime(init_time)} lead={step * 6}h "
                                    f"(amp={use_amp})"
                                )
                            native_sample = _from_oneforecast_longitude_order(
                                denorm[0, channels]
                            )
                            for lead_index in step_to_leads[step]:
                                pred[bi, lead_index] += native_sample / 4.0
                            print(
                                f"OneForecast init {global_i + 1}/{len(init_times)}: "
                                f"accumulated daily sample {step * 6}h",
                                flush=True,
                            )
                    torch.cuda.synchronize(device)
                    rollout_seconds += time.perf_counter() - t1
                    del x
                    torch.cuda.empty_cache()
                native = xr.Dataset(
                    {
                        "pred": (("init_time", "lead_time", "variable", "lat", "lon"), pred),
                        "valid_time": (("init_time", "lead_time"), make_valid_time(batch_init_times, lead_hours)),
                    },
                    coords={
                        "init_time": batch_init_times,
                        "lead_time": np.asarray(lead_hours, dtype=np.int32),
                        "variable": requested_variables,
                        "lat": native_lat_coord,
                        "lon": lon,
                    },
                )
                ds = native[["pred"]].sortby("lat", ascending=True).interp(
                    lat=target_lat[::-1],
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                ).sortby("lat", ascending=False)
                ds["valid_time"] = native["valid_time"]
                ds = ds.assign_coords(lat=target_lat, lon=lon)
                if first_write:
                    ds.attrs.update(_oneforecast_attrs(cfg, requested_variables, rollout_seconds, io_seconds, start_total, device))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    ds.to_zarr(out_path, mode="w", consolidated=False)
                    first_write = False
                else:
                    ds.to_zarr(out_path, mode="a", append_dim="init_time", consolidated=False)
                print(f"OneForecast wrote batch {batch_start + 1}-{batch_end}/{len(init_times)}", flush=True)
                del pred, native, ds
    import zarr

    zarr.consolidate_metadata(str(out_path))
    ds = xr.open_zarr(out_path, consolidated=True)
    ds.attrs.update(
        {
            "model_name": "oneforecast",
            "source_type": "official_inference",
            "source_uri": "https://huggingface.co/YuanGao-YG/OneForecast/tree/main",
            "missing_variables": ",".join([v for v in cfg["experiment"]["variables"] if v not in requested_variables]),
            "device_info": str(get_device_info()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "allow_synthetic": "false",
            "native_resolution": "120x240",
            "grid_resolution": "121x240",
            "required_resolution": "121x240",
            "resolution_note": "Official OneForecast inference uses canonical ERA5 initial states cropped to 120 latitude points; output is linearly interpolated/extrapolated to 121x240 for unified evaluation.",
            "longitude_order": (
                "WeatherBench2 0..358.5 rolled by +120 cells for the checkpoint's "
                "-180..178.5 order; predictions rolled back by -120 cells"
            ),
            "target_source": cfg["weatherbench2"]["era5_truth_uri"],
            "output_temporal_statistic": (
                "daily mean of forecast states at 00,06,12,18 UTC on each target day"
            ),
            "embedded_target": "omitted; canonical daily S2S truth is loaded during evaluation",
            "gpu_rollout_seconds": f"{rollout_seconds:.6f}",
            "input_load_seconds": f"{io_seconds:.6f}",
            "total_wall_seconds": f"{time.perf_counter() - start_total:.6f}",
            "cuda_peak_memory_gb": f"{torch.cuda.max_memory_allocated(device) / 1e9:.6f}",
        }
    )
    ds.to_zarr(out_path, mode="a", consolidated=True)
    profile = {
        "model_name": "oneforecast",
        "cache": output_path,
        "init_times": int(len(init_times)),
        "lead_hours": lead_hours,
        "variables": requested_variables,
        "input_load_seconds": io_seconds,
        "gpu_rollout_seconds": rollout_seconds,
        "total_wall_seconds": time.perf_counter() - start_total,
        "cuda_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    profile_path = Path("artifacts/profiles/oneforecast_build_cache.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return xr.open_zarr(out_path, consolidated=True)


def _oneforecast_attrs(cfg, requested_variables, rollout_seconds, io_seconds, start_total, device):
    import torch

    return {
        "model_name": "oneforecast",
        "source_type": "official_inference",
        "source_uri": "https://huggingface.co/YuanGao-YG/OneForecast/tree/main",
        "missing_variables": ",".join([v for v in cfg["experiment"]["variables"] if v not in requested_variables]),
        "device_info": str(get_device_info()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "allow_synthetic": "false",
        "native_resolution": "120x240",
        "grid_resolution": "121x240",
        "required_resolution": "121x240",
        "resolution_note": "Official OneForecast inference uses canonical ERA5 initial states at 1.5 degree longitude spacing, crops the native 120-latitude model input, and linearly interpolates/extrapolates output to the 121x240 protocol grid.",
        "longitude_order": (
            "WeatherBench2 0..358.5 rolled by +120 cells for the checkpoint's "
            "-180..178.5 order; predictions rolled back by -120 cells"
        ),
        "target_source": cfg["weatherbench2"]["era5_truth_uri"],
        "output_temporal_statistic": (
            "daily mean of forecast states at 00,06,12,18 UTC on each target day"
        ),
        "embedded_target": "omitted; canonical daily S2S truth is loaded during evaluation",
        "gpu_rollout_seconds": f"{rollout_seconds:.6f}",
        "input_load_seconds": f"{io_seconds:.6f}",
        "total_wall_seconds": f"{time.perf_counter() - start_total:.6f}",
        "cuda_peak_memory_gb": f"{torch.cuda.max_memory_allocated(device) / 1e9:.6f}",
    }
