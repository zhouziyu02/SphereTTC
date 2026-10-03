"""Official Huawei Pangu-Weather ONNX inference adapter."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.fourcastnetv2_official import (
    WB13_0P25_URI,
    _open_public_zarr,
    _specific_humidity_from_rh_temperature,
    _target_truth,
)
from src.utils.device import get_device_info
from src.utils.xarray_utils import normalize_lat_lon, open_zarr_lazy


PANGU_ROOT = Path("external/pangu_official")
PANGU_MODEL_24 = PANGU_ROOT / "pangu_weather_24.onnx"
PANGU_MODEL_6 = PANGU_ROOT / "pangu_weather_6.onnx"
PANGU_SOURCE_URI = "https://github.com/198808xc/Pangu-Weather"
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PANGU_UPPER = ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]
PANGU_SURFACE = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature"]


def _variables():
    from src.experiments.protocol import COMMON_EVALUATION_VARIABLES

    return list(COMMON_EVALUATION_VARIABLES)


def inspect_official():
    missing = []
    for checkpoint in (PANGU_MODEL_24, PANGU_MODEL_6):
        if not checkpoint.exists():
            missing.append(f"missing_{checkpoint}")
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            missing.append("missing_onnxruntime_cuda_provider")
    except Exception as exc:
        missing.append(f"missing_dependency:{type(exc).__name__}:{exc}")
    return {
        "status": "unavailable" if missing else "available_official_inference",
        "source_uri": PANGU_SOURCE_URI,
        "available_variables": _variables(),
        "available_leads": [24 * i for i in range(1, 11)],
        "missing_requirements": missing,
        "notes": (
            "Official Pangu 24-hour checkpoint advances target days; the official "
            "6-hour checkpoint supplies 06/12/18 UTC samples for daily means."
        ),
    }


def _init_times(cfg):
    year = int(cfg["experiment"]["year"])
    init_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00")
    frequency = int(cfg["experiment"].get("init_frequency_hours", 24))
    times = pd.date_range(f"{year}-01-01T00:00:00", init_end, freq=f"{frequency}h", inclusive="left")
    limit = cfg["experiment"].get("max_init_times")
    if limit is not None:
        times = times[: int(limit)]
    return times.values.astype("datetime64[ns]")


def _field(ds, name, when):
    import xarray as xr

    da = ds[name].sel(time=when)
    if "level" in da.dims:
        da = da.sel(level=PANGU_LEVELS).transpose("level", "lat", "lon")
    else:
        da = da.transpose("lat", "lon")
    return np.asarray(da.values, dtype=np.float32)


def _initial_state(ds, when):
    upper = []
    for name in PANGU_UPPER:
        try:
            upper.append(_field(ds, name, when))
        except KeyError:
            if name != "specific_humidity":
                raise
            rh = _field(ds, "relative_humidity", when)
            temp = _field(ds, "temperature", when)
            upper.append(np.stack([
                _specific_humidity_from_rh_temperature(rh[i], temp[i], level)
                for i, level in enumerate(PANGU_LEVELS)
            ]).astype(np.float32))
    surface = [_field(ds, name, when) for name in PANGU_SURFACE]
    return np.stack(upper).astype(np.float32), np.stack(surface).astype(np.float32)


def _session(checkpoint: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(checkpoint),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"Pangu ONNX session did not bind CUDAExecutionProvider: {checkpoint}")
    return session


def _interp(lead_values, names):
    import xarray as xr

    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 360.0 - 0.25, 1440, dtype=np.float32)
    target_lat = np.linspace(90.0, -90.0, 121, dtype=np.float32)
    target_lon = np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32)
    result = {}
    for name, values in zip(names, lead_values):
        da = xr.DataArray(
            np.asarray(values, dtype=np.float32),
            dims=("lead_time", "lat", "lon"),
            coords={"lead_time": [24, 72, 120, 168, 240], "lat": lat, "lon": lon},
        )
        result[name] = np.asarray(da.interp(lat=target_lat, lon=target_lon, method="linear").values, dtype=np.float32)
    return result


def build_pangu_cache(cfg: dict, output_path: str):
    import onnxruntime as ort
    import xarray as xr

    for checkpoint in (PANGU_MODEL_24, PANGU_MODEL_6):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    ort.set_default_logger_severity(3)
    input_data = _open_public_zarr(WB13_0P25_URI)
    input_data, _ = normalize_lat_lon(input_data)
    variables = _variables()
    leads = [int(x) for x in cfg["experiment"]["lead_time_hours"]]
    if leads != [24, 72, 120, 168, 240]:
        raise ValueError(f"Pangu adapter requires the strict protocol leads, got {leads}")
    init_times = _init_times(cfg)
    session_24 = _session(PANGU_MODEL_24)
    session_6 = _session(PANGU_MODEL_6)
    out_path = Path(output_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    first = True
    started = time.perf_counter()
    for index, init_time in enumerate(init_times):
        print(f"pangu init {index + 1}/{len(init_times)} {pd.to_datetime(init_time)}: loading official ERA5", flush=True)
        upper, surface = _initial_state(input_data, init_time)
        predictions = {}
        current_upper, current_surface = upper, surface
        for step in range(1, 11):
            current_upper, current_surface = session_24.run(
                None, {"input": current_upper, "input_surface": current_surface}
            )
            lead = step * 24
            if lead in leads:
                samples = [(current_upper, current_surface)]
                branch_upper, branch_surface = current_upper, current_surface
                for _ in range(3):
                    branch_upper, branch_surface = session_6.run(
                        None,
                        {"input": branch_upper, "input_surface": branch_surface},
                    )
                    samples.append((branch_upper, branch_surface))
                # Pangu output order is z,q,t,u,v over the 13 levels followed
                # by msl,u10,v10,t2m on the surface.
                native = {}
                for i, name in enumerate(PANGU_UPPER):
                    native[name] = np.mean(
                        np.stack([sample_upper[i : i + 1] for sample_upper, _ in samples]),
                        axis=0,
                    )
                for i, name in enumerate(PANGU_SURFACE):
                    native[name] = np.mean(
                        np.stack([sample_surface[i : i + 1] for _, sample_surface in samples]),
                        axis=0,
                    )
                predictions[lead] = native
                print(
                    f"pangu init {index + 1}/{len(init_times)}: "
                    f"saved daily mean for target lead {lead}h",
                    flush=True,
                )
        out_arrays = []
        lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
        lon = np.linspace(0.0, 360.0 - 0.25, 1440, dtype=np.float32)
        target_lat = np.linspace(90.0, -90.0, 121, dtype=np.float32)
        target_lon = np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32)
        for variable in variables:
            if variable[0] in "zqtuv" and variable[1:].isdigit() and int(variable[1:]) in PANGU_LEVELS:
                base = {"z": 0, "q": 1, "t": 2, "u": 3, "v": 4}[variable[0]]
                level_index = PANGU_LEVELS.index(int(variable[1:]))
                values = np.stack([predictions[h][PANGU_UPPER[base]][0, level_index] for h in leads])
            else:
                base_name = {"u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind", "t2m": "2m_temperature", "mslp": "mean_sea_level_pressure"}[variable]
                surface_index = PANGU_SURFACE.index(base_name)
                values = np.stack([predictions[h][PANGU_SURFACE[surface_index]][0] for h in leads])
            da = xr.DataArray(values, dims=("lead_time", "lat", "lon"), coords={"lead_time": leads, "lat": lat, "lon": lon})
            da = da.interp(lat=target_lat, lon=target_lon, method="linear")
            out_arrays.append(np.asarray(da.values, dtype=np.float32))
        pred = np.stack(out_arrays, axis=1)
        ds = xr.Dataset(
            {
                "pred": (("init_time", "lead_time", "variable", "lat", "lon"), pred[None]),
                "valid_time": (("init_time", "lead_time"), np.asarray([[init_time + np.timedelta64(h, "h") for h in leads]], dtype="datetime64[ns]")),
            },
            coords={
                "init_time": [init_time],
                "lead_time": leads,
                "variable": variables,
                "lat": target_lat,
                "lon": target_lon,
            },
            attrs={
                "model_name": "pangu",
                "source_type": "official_inference",
                "source_uri": PANGU_SOURCE_URI,
                "checkpoint_24h": str(PANGU_MODEL_24),
                "checkpoint_6h": str(PANGU_MODEL_6),
                "initial_condition_uri": WB13_0P25_URI,
                "target_source": cfg["weatherbench2"]["era5_truth_uri"],
                "protocol_variables": json.dumps(variables),
                "allow_synthetic": "false",
                "grid_resolution": "121x240",
                "native_resolution": "721x1440",
                "official_native_step_hours": "24 with 6-hour sub-daily sampling",
                "output_temporal_statistic": (
                    "daily mean of forecast states at 00,06,12,18 UTC on each target day"
                ),
                "embedded_target": "omitted; canonical daily S2S truth is loaded during evaluation",
                "device_info": str(get_device_info()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_wall_seconds": f"{time.perf_counter() - started:.6f}",
            },
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(out_path, mode="w" if first else "a", append_dim=None if first else "init_time", consolidated=False)
        first = False
        print(f"pangu wrote init {index + 1}/{len(init_times)}", flush=True)
        del upper, surface, current_upper, current_surface, predictions, pred, ds
    import zarr

    zarr.consolidate_metadata(str(out_path))
    return xr.open_zarr(out_path, consolidated=True)
