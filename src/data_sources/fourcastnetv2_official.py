from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data_sources.daily_mean import six_hour_step_to_lead_indices
from src.utils.device import get_device_info
from src.utils.time import make_valid_time
from src.utils.xarray_utils import PRESSURE_LEVELS_13, normalize_lat_lon, open_zarr_lazy, select_canonical_dataarray, time_dim


WB13_0P25_URI = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
ARCO_U100_URI = "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2"

FCNV2_ORDERING = [
    "10u", "10v", "100u", "100v", "2t", "sp", "msl", "tcwv",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000",
    "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000",
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000",
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000",
    "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700", "r850", "r925", "r1000",
]

OUTPUT_CHANNELS = {
    **{f"z{level}": FCNV2_ORDERING.index(f"z{level}") for level in PRESSURE_LEVELS_13},
    **{f"t{level}": FCNV2_ORDERING.index(f"t{level}") for level in PRESSURE_LEVELS_13},
    **{f"u{level}": FCNV2_ORDERING.index(f"u{level}") for level in PRESSURE_LEVELS_13},
    **{f"v{level}": FCNV2_ORDERING.index(f"v{level}") for level in PRESSURE_LEVELS_13},
    "u10": FCNV2_ORDERING.index("10u"),
    "v10": FCNV2_ORDERING.index("10v"),
    "t2m": FCNV2_ORDERING.index("2t"),
    "mslp": FCNV2_ORDERING.index("msl"),
}
for _level in PRESSURE_LEVELS_13:
    OUTPUT_CHANNELS[f"q{_level}"] = -1

WB_SURFACE = {
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "2t": "2m_temperature",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "tcwv": "total_column_water_vapour",
}

WB_LEVEL = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t": "temperature",
    "r": "relative_humidity",
}


def inspect_official(root: str = "external/ai-models-fourcastnetv2"):
    missing = []
    root_path = Path(root)
    if not root_path.exists():
        missing.append("missing_official_repo")
    for rel in ["assets/weights.tar", "assets/global_means.npy", "assets/global_stds.npy"]:
        if not (root_path / rel).exists():
            missing.append(f"missing_{rel}")
    try:
        import ai_models_fourcastnetv2  # noqa: F401
        import torch_harmonics  # noqa: F401
    except Exception as exc:
        missing.append(f"missing_dependency:{type(exc).__name__}:{exc}")
    return {
        "status": "unavailable" if missing else "available_official_inference",
        "source_uri": "https://github.com/ecmwf-lab/ai-models-fourcastnetv2",
        "available_variables": list(OUTPUT_CHANNELS),
        "available_leads": [6 * i for i in range(1, 41)],
        "available_times": {"count": 366 * 4, "start": "2020-01-01T00:00:00", "end": "2020-12-31T18:00:00"},
        "missing_requirements": missing,
        "notes": (
            "Uses official checkpoint/statistics. Initial conditions are built from real WeatherBench2 "
            "0.25 degree ERA5 plus real ARCO ERA5 u100/v100 remapped from reduced Gaussian grid."
        ),
    }


def _load_model(root: Path, device):
    import torch

    sys.path.insert(0, str(root.resolve()))
    from ai_models_fourcastnetv2.fourcastnetv2.sfnonet import FourierNeuralOperatorNet

    model = FourierNeuralOperatorNet().to(device).eval()
    checkpoint = torch.load(root / "assets/weights.tar", map_location=device, weights_only=False)
    state = {}
    for key, val in checkpoint["model_state"].items():
        name = key[7:] if key.startswith("module.") else key
        if name != "ged":
            state[name] = val
    model.load_state_dict(state, strict=False)
    return model


def _open_public_zarr(uri: str):
    if uri.startswith("gs://"):
        import gcsfs
        import xarray as xr

        fs = gcsfs.GCSFileSystem(token="anon")
        return xr.open_zarr(fs.get_mapper(uri[5:]), consolidated=True, chunks=None)
    return open_zarr_lazy(uri)


def _regular_grid():
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 360.0 - 0.25, 1440, dtype=np.float32)
    return lat, lon


def _unit_sphere(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    clat = np.cos(lat)
    return np.column_stack((clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)))


def _u100_nearest_index(arco, cache_path: Path):
    if cache_path.exists():
        return np.load(cache_path)["index"]
    from scipy.spatial import cKDTree

    lat, lon = _regular_grid()
    target_lon, target_lat = np.meshgrid(lon, lat)
    source_xyz = _unit_sphere(arco["latitude"].values, arco["longitude"].values)
    target_xyz = _unit_sphere(target_lat.ravel(), target_lon.ravel())
    _, index = cKDTree(source_xyz).query(target_xyz, k=1, workers=-1)
    index = index.astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, index=index)
    return index


def _channel_from_weatherbench(ds, short_name: str, when: np.datetime64):
    if short_name in WB_SURFACE:
        arr = ds[WB_SURFACE[short_name]].sel(time=when).values
        return np.asarray(arr, dtype=np.float32)
    prefix = short_name[0]
    level = int(short_name[1:])
    arr = ds[WB_LEVEL[prefix]].sel(time=when, level=level).values
    return np.asarray(arr, dtype=np.float32)


def _u100_channels(arco, nn_index, when: np.datetime64):
    u = np.asarray(arco["u100"].sel(time=when).values, dtype=np.float32)[nn_index].reshape(721, 1440)
    v = np.asarray(arco["v100"].sel(time=when).values, dtype=np.float32)[nn_index].reshape(721, 1440)
    return u, v


def _load_initial_state(wb, arco, nn_index, when: np.datetime64):
    channels = []
    u100 = v100 = None
    for name in FCNV2_ORDERING:
        if name == "100u":
            if u100 is None:
                u100, v100 = _u100_channels(arco, nn_index, when)
            channels.append(u100)
        elif name == "100v":
            if v100 is None:
                u100, v100 = _u100_channels(arco, nn_index, when)
            channels.append(v100)
        else:
            channels.append(_channel_from_weatherbench(wb, name, when))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _interp_output_to_121x240(native, variables):
    import xarray as xr

    lat, lon = _regular_grid()
    target_lat = np.linspace(90.0, -90.0, 121, dtype=np.float32)
    target_lon = np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32)
    da = xr.DataArray(
        native,
        dims=("variable", "lat", "lon"),
        coords={"variable": variables, "lat": lat, "lon": lon},
    )
    out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    return np.asarray(out.values, dtype=np.float32)


def _specific_humidity_from_rh_temperature(rh_percent, temperature_k, level_hpa):
    epsilon = 0.622
    temp_c = temperature_k - 273.15
    saturation_vapor_pressure = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    vapor_pressure = np.clip(rh_percent, 0.0, 100.0) / 100.0 * saturation_vapor_pressure
    pressure_pa = float(level_hpa) * 100.0
    return epsilon * vapor_pressure / (pressure_pa - (1.0 - epsilon) * vapor_pressure)


def _extract_eval_native(all_channels, variables):
    arrays = []
    for variable in variables:
        if variable.startswith("q") and variable[1:].isdigit():
            level = int(variable[1:])
            rh = all_channels[FCNV2_ORDERING.index(f"r{level}")]
            temp = all_channels[FCNV2_ORDERING.index(f"t{level}")]
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
    target = target.transpose("init_time", "lead_time", "variable", "lat", "lon")
    return target


def build_fourcastnetv2_cache(cfg: dict, output_path: str):
    import pandas as pd
    import torch
    import xarray as xr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FourCastNetV2 official inference.")
    start_total = time.perf_counter()
    device = torch.device("cuda")
    root = Path("external/ai-models-fourcastnetv2")
    wb = _open_public_zarr(WB13_0P25_URI)
    wb, _ = normalize_lat_lon(wb)
    arco = _open_public_zarr(ARCO_U100_URI)
    nn_index = _u100_nearest_index(arco, Path("artifacts/local_data_cache/fourcastnetv2/u100_reduced_gg_to_0p25_index.npz"))

    requested_variables = [v for v in cfg["experiment"]["variables"] if v in OUTPUT_CHANNELS]
    if not requested_variables:
        raise RuntimeError("No requested variables are available in FourCastNetV2 output channels.")
    lead_hours = [int(x) for x in cfg["experiment"]["lead_time_hours"]]
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

    model = _load_model(root, device)
    means = np.load(root / "assets/global_means.npy").astype(np.float32)[:, : len(FCNV2_ORDERING)]
    stds = np.load(root / "assets/global_stds.npy").astype(np.float32)[:, : len(FCNV2_ORDERING)]
    load_seconds = 0.0
    rollout_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    out_path = Path(output_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    batch_size = int(cfg["experiment"].get("official_inference_batch_size", 8))
    first_write = True
    with torch.no_grad():
        for batch_start in range(0, len(init_times), batch_size):
            batch_end = min(batch_start + batch_size, len(init_times))
            batch_init_times = init_times[batch_start:batch_end]
            pred = np.zeros((len(batch_init_times), len(lead_hours), len(requested_variables), 121, 240), dtype=np.float32)
            print(f"FourCastNetV2 batch {batch_start + 1}-{batch_end}/{len(init_times)}", flush=True)
            for bi, init_time in enumerate(batch_init_times):
                global_i = batch_start + bi
                print(f"FourCastNetV2 init {global_i + 1}/{len(init_times)} {pd.to_datetime(init_time)}: loading ERA5", flush=True)
                t0 = time.perf_counter()
                state = _load_initial_state(wb, arco, nn_index, init_time)
                load_seconds += time.perf_counter() - t0
                norm = (state[None] - means) / stds
                x = torch.as_tensor(norm, dtype=torch.float32, device=device)
                print(f"FourCastNetV2 init {global_i + 1}/{len(init_times)}: rollout {max_step} steps", flush=True)
                t1 = time.perf_counter()
                for step in range(1, max_step + 1):
                    x = model(x)
                    if step in step_to_leads:
                        denorm = x.detach().cpu().numpy() * stds + means
                        native = _extract_eval_native(denorm[0], requested_variables)
                        downsampled = _interp_output_to_121x240(native, requested_variables)
                        for lead_index in step_to_leads[step]:
                            pred[bi, lead_index] += downsampled / 4.0
                        print(
                            f"FourCastNetV2 init {global_i + 1}/{len(init_times)}: "
                            f"accumulated daily sample {step * 6}h",
                            flush=True,
                        )
                torch.cuda.synchronize(device)
                rollout_seconds += time.perf_counter() - t1
                del x
                torch.cuda.empty_cache()

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
                    "model_name": "fourcastnetv2",
                    "source_type": "official_inference",
                    "source_uri": "https://github.com/ecmwf-lab/ai-models-fourcastnetv2",
                    "initial_condition_uri": WB13_0P25_URI,
                    "u100_v100_uri": ARCO_U100_URI,
                    "u100_v100_remap": "nearest-neighbor reduced Gaussian to 0.25 degree regular lat/lon using spherical KDTree",
                    "missing_variables": ",".join([v for v in cfg["experiment"]["variables"] if v not in requested_variables]),
                    "device_info": str(get_device_info()),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "allow_synthetic": "false",
                    "native_resolution": "721x1440",
                    "grid_resolution": "121x240",
                    "longitude_cyclic_point_added": "false",
                    "output_temporal_statistic": (
                        "daily mean of forecast states at 00,06,12,18 UTC on each target day"
                    ),
                    "embedded_target": "omitted; canonical daily S2S truth is loaded during evaluation",
                    "gpu_rollout_seconds": f"{rollout_seconds:.6f}",
                    "input_load_seconds": f"{load_seconds:.6f}",
                    "total_wall_seconds": f"{time.perf_counter() - start_total:.6f}",
                    "cuda_peak_memory_gb": f"{torch.cuda.max_memory_allocated(device) / 1e9:.6f}",
                },
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if first_write:
                ds.to_zarr(out_path, mode="w", consolidated=False)
                first_write = False
            else:
                ds.to_zarr(out_path, mode="a", append_dim="init_time", consolidated=False)
            print(f"FourCastNetV2 wrote batch {batch_start + 1}-{batch_end}/{len(init_times)}", flush=True)
            del pred, ds

    import zarr

    zarr.consolidate_metadata(str(out_path))
    ds = xr.open_zarr(out_path, consolidated=True)
    ds.attrs.update(
        {
            "gpu_rollout_seconds": f"{rollout_seconds:.6f}",
            "input_load_seconds": f"{load_seconds:.6f}",
            "total_wall_seconds": f"{time.perf_counter() - start_total:.6f}",
            "cuda_peak_memory_gb": f"{torch.cuda.max_memory_allocated(device) / 1e9:.6f}",
        }
    )
    ds.to_zarr(out_path, mode="a", consolidated=True)
    profile = {
        "model_name": "fourcastnetv2",
        "cache": output_path,
        "init_times": int(len(init_times)),
        "lead_hours": lead_hours,
        "variables": requested_variables,
        "input_load_seconds": load_seconds,
        "gpu_rollout_seconds": rollout_seconds,
        "total_wall_seconds": time.perf_counter() - start_total,
        "cuda_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    profile_path = Path("artifacts/profiles/fourcastnetv2_build_cache.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return xr.open_zarr(out_path, consolidated=True)
