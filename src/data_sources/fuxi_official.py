from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.daily_mean import six_hour_step_to_lead_indices
from src.data_sources.fourcastnetv2_official import WB13_0P25_URI, _open_public_zarr, _specific_humidity_from_rh_temperature
from src.utils.device import get_device_info
from src.utils.time import make_valid_time
from src.utils.xarray_utils import PRESSURE_LEVELS_10, normalize_lat_lon, open_zarr_lazy, select_canonical_dataarray, time_dim


FUXI_ROOT = Path("external/FuXi_EC")
FUXI_SOURCE_URI = "https://github.com/tpys/FuXi; https://zenodo.org/records/10401602"
FUXI_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
FUXI_PL_NAMES = ["z", "t", "u", "v", "r"]
FUXI_SFC_NAMES = ["t2m", "u10", "v10", "msl", "tp"]
FUXI_CHANNELS = [f"{name}{level}" for name in FUXI_PL_NAMES for level in FUXI_LEVELS]
FUXI_CHANNELS += FUXI_SFC_NAMES

WB_LEVEL = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "r": "relative_humidity",
}
WB_SURFACE = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
TP_CANDIDATES = ["total_precipitation_6hr", "total_precipitation", "tp"]

OUTPUT_CHANNELS = {
    # The official FuXi tensor contains all 13 pressure levels.  The old
    # adapter accidentally used PRESSURE_LEVELS_10 here, silently dropping
    # 50/400/600 hPa and making the result incompatible with the current
    # strict-69 protocol.
    **{f"z{level}": FUXI_CHANNELS.index(f"z{level}") for level in FUXI_LEVELS},
    **{f"t{level}": FUXI_CHANNELS.index(f"t{level}") for level in FUXI_LEVELS},
    **{f"u{level}": FUXI_CHANNELS.index(f"u{level}") for level in FUXI_LEVELS},
    **{f"v{level}": FUXI_CHANNELS.index(f"v{level}") for level in FUXI_LEVELS},
    "u10": FUXI_CHANNELS.index("u10"),
    "v10": FUXI_CHANNELS.index("v10"),
    "t2m": FUXI_CHANNELS.index("t2m"),
    "mslp": FUXI_CHANNELS.index("msl"),
}
for _level in FUXI_LEVELS:
    OUTPUT_CHANNELS[f"q{_level}"] = -1


def inspect_official(root: str = str(FUXI_ROOT)):
    missing = []
    root_path = Path(root)
    if not root_path.exists():
        missing.append("missing_official_weights_dir")
    for rel in ["short.onnx", "short", "medium.onnx", "medium", "long.onnx", "long"]:
        if not (root_path / rel).exists():
            missing.append(f"missing_{rel}")
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            missing.append("missing_onnxruntime_cuda_provider")
    except Exception as exc:
        missing.append(f"missing_dependency:{type(exc).__name__}:{exc}")
    return {
        "status": "unavailable" if missing else "available_official_inference",
        "source_uri": FUXI_SOURCE_URI,
        "available_variables": list(OUTPUT_CHANNELS),
        "available_leads": [6 * i for i in range(1, 41)],
        "available_times": {"count": 366 * 4, "start": "2020-01-01T00:00:00", "end": "2020-12-31T18:00:00"},
        "missing_requirements": missing,
        "notes": (
            "Uses official FuXi ONNX checkpoints. Initial conditions are built from real WeatherBench2 "
            "0.25 degree ERA5 WB13 data. Specific humidity is derived from FuXi relative humidity and temperature."
        ),
    }


def _session_options():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    return options


def _load_session(path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path),
        sess_options=_session_options(),
        providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"FuXi ONNX session did not bind CUDAExecutionProvider: {path}")
    return session


def _time_encoding(init_time, total_step: int, freq: int = 6):
    init_time = np.array([pd.to_datetime(init_time)])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        periods = [pd.Period(t, "h") for t in times.reshape(-1)]
        values = np.array([(p.day_of_year / 366, p.hour / 24) for p in periods], dtype=np.float32)
        temb = np.concatenate([np.sin(values), np.cos(values)], axis=-1).reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)


def _regular_grid():
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 360.0 - 0.25, 1440, dtype=np.float32)
    return lat, lon


def _tp_to_fuxi_mm(ds, when: np.datetime64):
    name = next((candidate for candidate in TP_CANDIDATES if candidate in ds.data_vars), None)
    if name is None:
        raise KeyError(f"FuXi requires real precipitation input; none of {TP_CANDIDATES} found in WB13 dataset.")
    arr = np.asarray(ds[name].sel(time=when).values, dtype=np.float32)
    units = str(ds[name].attrs.get("units", "")).lower()
    if units in {"m", "meter", "metre", "meters", "metres"} or np.nanmax(arr) < 10.0:
        arr = arr * 1000.0
    return np.clip(arr, 0.0, 1000.0).astype(np.float32)


def _channel_from_weatherbench(ds, channel: str, when: np.datetime64):
    if channel == "tp":
        return _tp_to_fuxi_mm(ds, when)
    if channel in WB_SURFACE:
        return np.asarray(ds[WB_SURFACE[channel]].sel(time=when).values, dtype=np.float32)
    prefix = channel[0]
    level = int(channel[1:])
    return np.asarray(ds[WB_LEVEL[prefix]].sel(time=when, level=level).values, dtype=np.float32)


def _relative_humidity_fraction_to_percent(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32) * 100.0, 0.0, 100.0)


def _load_initial_state(wb, init_time: np.datetime64):
    prev_time = init_time - np.timedelta64(6, "h")
    times = np.asarray([prev_time, init_time], dtype="datetime64[ns]")
    groups = []
    for name in FUXI_PL_NAMES:
        values = wb[WB_LEVEL[name]].sel(time=times, level=FUXI_LEVELS).values
        # WeatherBench2's derived relative_humidity is a fraction (typically
        # 0..1), while the official FuXi ERA5 preprocessing and ONNX checkpoint
        # use relative humidity in percent (0..100).
        if name == "r":
            values = _relative_humidity_fraction_to_percent(values)
        groups.append(np.asarray(values, dtype=np.float32))
    surface_groups = []
    for name in ["t2m", "u10", "v10", "msl"]:
        values = wb[WB_SURFACE[name]].sel(time=times).values[:, None]
        surface_groups.append(np.asarray(values, dtype=np.float32))
    tp = np.stack([_tp_to_fuxi_mm(wb, when) for when in times], axis=0)[:, None]
    surface_groups.append(tp.astype(np.float32, copy=False))
    return np.concatenate(groups + surface_groups, axis=1).astype(np.float32, copy=False)


def _interp_output_to_121x240(native, variables):
    import xarray as xr

    lat, lon = _regular_grid()
    target_lat = np.linspace(90.0, -90.0, 121, dtype=np.float32)
    target_lon = np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32)
    da = xr.DataArray(native, dims=("variable", "lat", "lon"), coords={"variable": variables, "lat": lat, "lon": lon})
    return np.asarray(da.interp(lat=target_lat, lon=target_lon, method="linear").values, dtype=np.float32)


def _extract_eval_native(all_channels, variables):
    arrays = []
    for variable in variables:
        if variable.startswith("q") and variable[1:].isdigit():
            level = int(variable[1:])
            rh = all_channels[FUXI_CHANNELS.index(f"r{level}")]
            temp = all_channels[FUXI_CHANNELS.index(f"t{level}")]
            arrays.append(_specific_humidity_from_rh_temperature(rh, temp, level).astype(np.float32))
        else:
            arrays.append(all_channels[OUTPUT_CHANNELS[variable]])
    return np.stack(arrays, axis=0).astype(np.float32)


def _target_truth(truth, variables, init_times, lead_hours):
    import xarray as xr

    valid = make_valid_time(init_times, lead_hours)
    arrays = []
    for var in variables:
        base = select_canonical_dataarray(truth, var)
        slices = []
        for lead_idx in range(len(lead_hours)):
            vt = valid[:, lead_idx]
            part = base.sel({time_dim(truth): vt}).rename({time_dim(truth): "init_time"})
            part = part.assign_coords(init_time=init_times)
            slices.append(part)
        da = xr.concat(slices, dim="lead_time")
        da = da.assign_coords(lead_time=np.asarray(lead_hours, dtype=np.int32), init_time=init_times)
        arrays.append(da.transpose("init_time", "lead_time", "lat", "lon").astype("float32"))
    target = xr.concat(arrays, dim="variable", coords="minimal", compat="override").assign_coords(variable=variables)
    return target.transpose("init_time", "lead_time", "variable", "lat", "lon")


def _stage_for_step(step: int) -> str:
    if step <= 20:
        return "short"
    if step <= 40:
        return "medium"
    return "long"


def _gpu_memory_gb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        return None
    return None


def build_fuxi_cache(cfg: dict, output_path: str):
    import onnxruntime as ort
    import xarray as xr

    start_total = time.perf_counter()
    ort.set_default_logger_severity(3)
    wb = _open_public_zarr(WB13_0P25_URI)
    wb, _ = normalize_lat_lon(wb)
    requested_variables = [v for v in cfg["experiment"]["variables"] if v in OUTPUT_CHANNELS]
    missing_requested = [v for v in cfg["experiment"]["variables"] if v not in requested_variables]
    if missing_requested and cfg["experiment"].get("strict_real_data", True):
        raise RuntimeError(f"FuXi cannot provide requested variables: {missing_requested}")

    lead_hours = [int(x) for x in cfg["experiment"]["lead_time_hours"]]
    if any(h % 6 for h in lead_hours):
        raise RuntimeError(f"FuXi supports 6-hourly lead times only: {lead_hours}")
    step_to_leads = six_hour_step_to_lead_indices(lead_hours)
    max_step = max(step_to_leads)

    year = int(cfg["experiment"]["year"])
    init_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00")
    frequency = int(cfg["experiment"].get("init_frequency_hours", 24))
    all_times = pd.date_range(f"{year}-01-01T00:00:00", init_end, freq=f"{frequency}h", inclusive="left")
    max_init = cfg["experiment"].get("max_init_times")
    if max_init is not None:
        all_times = all_times[: int(max_init)]
    init_times = all_times.values.astype("datetime64[ns]")

    sessions = {}
    load_seconds = 0.0
    rollout_seconds = 0.0
    out_path = Path(output_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    batch_size = int(cfg["experiment"].get("official_inference_batch_size", 8))
    first_write = True

    for batch_start in range(0, len(init_times), batch_size):
        batch_end = min(batch_start + batch_size, len(init_times))
        batch_init_times = init_times[batch_start:batch_end]
        pred = np.zeros((len(batch_init_times), len(lead_hours), len(requested_variables), 121, 240), dtype=np.float32)
        print(f"FuXi batch {batch_start + 1}-{batch_end}/{len(init_times)}", flush=True)
        for bi, init_time in enumerate(batch_init_times):
            global_i = batch_start + bi
            print(f"FuXi init {global_i + 1}/{len(init_times)} {pd.to_datetime(init_time)}: loading ERA5", flush=True)
            t0 = time.perf_counter()
            state = _load_initial_state(wb, init_time)
            load_seconds += time.perf_counter() - t0
            inputs = state[None]
            tembs = _time_encoding(pd.to_datetime(init_time), max_step)
            print(f"FuXi init {global_i + 1}/{len(init_times)}: rollout {max_step} steps", flush=True)
            t1 = time.perf_counter()
            for step in range(1, max_step + 1):
                stage = _stage_for_step(step)
                if stage not in sessions:
                    print(f"FuXi loading {stage} ONNX", flush=True)
                    sessions[stage] = _load_session(FUXI_ROOT / f"{stage}.onnx")
                output_name = sessions[stage].get_outputs()[0].name
                inputs = sessions[stage].run([output_name], {"input": inputs, "temb": tembs[step - 1]})[0]
                if step in step_to_leads:
                    native = _extract_eval_native(inputs[0, -1], requested_variables)
                    downsampled = _interp_output_to_121x240(native, requested_variables)
                    for lead_index in step_to_leads[step]:
                        pred[bi, lead_index] += downsampled / 4.0
                    print(
                        f"FuXi init {global_i + 1}/{len(init_times)}: "
                        f"accumulated daily sample {step * 6}h",
                        flush=True,
                    )
            rollout_seconds += time.perf_counter() - t1

        ds = xr.Dataset(
            {
                "pred": (("init_time", "lead_time", "variable", "lat", "lon"), pred),
                "valid_time": (("init_time", "lead_time"), make_valid_time(batch_init_times, lead_hours)),
            },
            coords={
                "init_time": batch_init_times,
                "lead_time": np.asarray(lead_hours, dtype=np.int32),
                "variable": requested_variables,
                "lat": np.linspace(90.0, -90.0, 121, dtype=np.float32),
                "lon": np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32),
            },
            attrs={
                "model_name": "fuxi",
                "source_type": "official_inference",
                "source_uri": FUXI_SOURCE_URI,
                "initial_condition_uri": WB13_0P25_URI,
                "missing_variables": ",".join(missing_requested),
                "device_info": str(get_device_info()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "allow_synthetic": "false",
                "native_resolution": "721x1440",
                "grid_resolution": "121x240",
                "longitude_cyclic_point_added": "false",
                "q_conversion": "specific humidity derived from relative humidity and temperature at each pressure level",
                "relative_humidity_input_units": (
                    "percent; WeatherBench2 fractional relative_humidity multiplied by 100"
                ),
                "output_temporal_statistic": (
                    "daily mean of forecast states at 00,06,12,18 UTC on each target day"
                ),
                "embedded_target": "omitted; canonical daily S2S truth is loaded during evaluation",
                "gpu_rollout_seconds": f"{rollout_seconds:.6f}",
                "input_load_seconds": f"{load_seconds:.6f}",
                "total_wall_seconds": f"{time.perf_counter() - start_total:.6f}",
                "cuda_peak_memory_gb": "" if _gpu_memory_gb() is None else f"{_gpu_memory_gb():.6f}",
            },
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if first_write:
            ds.to_zarr(out_path, mode="w", consolidated=False)
            first_write = False
        else:
            ds.to_zarr(out_path, mode="a", append_dim="init_time", consolidated=False)
        print(f"FuXi wrote batch {batch_start + 1}-{batch_end}/{len(init_times)}", flush=True)
        del pred, ds

    import zarr

    zarr.consolidate_metadata(str(out_path))
    ds = xr.open_zarr(out_path, consolidated=True)
    ds.attrs.update(
        {
            "gpu_rollout_seconds": f"{rollout_seconds:.6f}",
            "input_load_seconds": f"{load_seconds:.6f}",
            "total_wall_seconds": f"{time.perf_counter() - start_total:.6f}",
            "cuda_peak_memory_gb": "" if _gpu_memory_gb() is None else f"{_gpu_memory_gb():.6f}",
        }
    )
    ds.to_zarr(out_path, mode="a", consolidated=True)
    profile = {
        "model_name": "fuxi",
        "cache": output_path,
        "init_times": int(len(init_times)),
        "lead_hours": lead_hours,
        "variables": requested_variables,
        "input_load_seconds": load_seconds,
        "gpu_rollout_seconds": rollout_seconds,
        "total_wall_seconds": time.perf_counter() - start_total,
        "cuda_peak_memory_gb": _gpu_memory_gb(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    profile_path = Path("artifacts/profiles/fuxi_build_cache.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return xr.open_zarr(out_path, consolidated=True)
