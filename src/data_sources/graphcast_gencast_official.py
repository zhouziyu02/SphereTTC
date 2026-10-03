"""Official GraphCast and GenCast inference adapters.

This module intentionally uses the DeepMind reference implementation and
checkpoint format.  It does not fall back to an archive forecast or a
synthetic field when an official asset is missing.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources.daily_mean import daily_mean_sample_hours
from src.data_sources.fourcastnetv2_official import (
    WB13_0P25_URI,
    _open_public_zarr,
    _target_truth,
)
from src.utils.device import get_device_info
from src.utils.xarray_utils import normalize_lat_lon, open_zarr_lazy


GRAPHCAST_CHECKPOINT = Path("external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz")
GENCAST_CHECKPOINT = Path("external/official_checkpoints/gencast/GenCast_1p0deg_2019.npz")
GRAPHCAST_STATS = Path("external/official_checkpoints/graphcast/stats")
GENCAST_STATS = Path("external/official_checkpoints/gencast/stats")
GRAPHCAST_SOURCE_URI = "https://github.com/google-deepmind/graphcast"
GENCAST_SOURCE_URI = "https://github.com/google-deepmind/graphcast"


def _variables():
    from src.experiments.protocol import COMMON_EVALUATION_VARIABLES

    return list(COMMON_EVALUATION_VARIABLES)


def inspect_graphcast_official():
    missing = []
    for path in [GRAPHCAST_CHECKPOINT, *GRAPHCAST_STATS.glob("*.nc")]:
        if not path.exists():
            missing.append(f"missing_{path}")
    try:
        import haiku  # noqa: F401
        import jax  # noqa: F401
        import trimesh  # noqa: F401
        from graphcast import graphcast  # noqa: F401
    except Exception as exc:
        missing.append(f"missing_dependency:{type(exc).__name__}:{exc}")
    return {
        "status": "unavailable" if missing else "available_official_inference",
        "source_uri": GRAPHCAST_SOURCE_URI,
        "available_variables": _variables(),
        "available_leads": [6 * i for i in range(1, 41)],
        "missing_requirements": missing,
        "notes": "Official DeepMind GraphCast_small checkpoint; 1 degree ERA5 input and 6-hour autoregressive rollout. Output is remapped to the frozen 121x240 evaluation grid.",
    }


def inspect_gencast_official():
    missing = []
    for path in [GENCAST_CHECKPOINT, *GENCAST_STATS.glob("*.nc")]:
        if not path.exists():
            missing.append(f"missing_{path}")
    try:
        import haiku  # noqa: F401
        import jax  # noqa: F401
        import trimesh  # noqa: F401
        from graphcast import gencast  # noqa: F401
    except Exception as exc:
        missing.append(f"missing_dependency:{type(exc).__name__}:{exc}")
    return {
        "status": "unavailable" if missing else "available_official_inference",
        "source_uri": GENCAST_SOURCE_URI,
        "available_variables": _variables(),
        "available_leads": [12 * i for i in range(1, 21)],
        "missing_requirements": missing,
        "notes": "Official DeepMind GenCast 1 degree checkpoint; 12-hour autoregressive rollout and fixed seeded ensemble sampling. Output is remapped to the frozen 121x240 evaluation grid.",
    }


def _open_truth_and_input(cfg):
    truth = open_zarr_lazy(cfg["weatherbench2"]["era5_truth_uri"])
    truth, _ = normalize_lat_lon(truth)
    input_data = _open_public_zarr(WB13_0P25_URI)
    input_data, _ = normalize_lat_lon(input_data)
    # The official small checkpoints are 1 degree models.  Select the exact
    # 1-degree sub-grid from the official 0.25-degree ERA5 source rather than
    # using an archive forecast or synthetic downsampling.
    input_data = input_data.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
    return truth, input_data


def _init_times(cfg):
    year = int(cfg["experiment"]["year"])
    init_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00")
    frequency = int(cfg["experiment"].get("init_frequency_hours", 24))
    times = pd.date_range(f"{year}-01-01T00:00:00", init_end, freq=f"{frequency}h", inclusive="left")
    limit = cfg["experiment"].get("max_init_times")
    if limit is not None:
        times = times[: int(limit)]
    return times.values.astype("datetime64[ns]")


def _model_input_dataset(input_data, init_time, max_lead, task_config):
    import xarray as xr

    start = pd.Timestamp(init_time) - pd.Timedelta(task_config.input_duration)
    end = pd.Timestamp(init_time) + pd.Timedelta(hours=max_lead)
    data = input_data.sel(time=slice(start, end))
    if task_config.input_duration == "24h":
        # GenCast's official task is a 12-hour cadence model.  The source ERA5
        # is 6-hourly, so retain the exact init-24h, init-12h, ... sequence
        # rather than allowing extract_inputs_targets_forcings to feed four
        # 6-hour frames into a two-frame checkpoint.
        data = data.isel(time=slice(0, None, 2))
    needed = set(task_config.input_variables) | set(task_config.target_variables) | set(task_config.forcing_variables)
    needed -= {"toa_incident_solar_radiation", "year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos"}
    available = [name for name in needed if name in data.data_vars]
    data = data[available].transpose("time", "level", "lat", "lon", missing_dims="ignore")
    data = data.assign_coords(datetime=("time", data.time.values))
    # The reference utility computes the derived solar/progress forcings and
    # keeps the exact checkpoint variable names/order.
    from graphcast import data_utils

    lead_step = 6 if task_config.input_duration == "12h" else 12
    native_leads = [f"{lead_step * i}h" for i in range(1, max_lead // lead_step + 1)]
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=native_leads,
        **dataclasses.asdict(task_config),
    )
    # The reference notebooks use a singleton batch dimension even for one
    # forecast.  Keeping it here is required by model_utils.dataset_to_stacked.
    return (
        inputs.expand_dims(batch=[0]),
        targets.expand_dims(batch=[0]),
        forcings.expand_dims(batch=[0]),
    )


def _stats(root: Path):
    import xarray as xr

    return {
        name: xr.open_dataset(root / f"{name}.nc")
        for name in ["diffs_stddev_by_level", "mean_by_level", "stddev_by_level"]
        if (root / f"{name}.nc").exists()
    }


def _graphcast_predictor(ckpt):
    import functools
    import haiku as hk
    import jax
    from graphcast import autoregressive, casting, graphcast, normalization, rollout

    stats = _stats(GRAPHCAST_STATS)
    diffs = stats["diffs_stddev_by_level"]
    mean = stats["mean_by_level"]
    std = stats["stddev_by_level"]

    def construct():
        predictor = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs,
            mean_by_level=mean,
            stddev_by_level=std,
        )
        return autoregressive.Predictor(predictor, gradient_checkpointing=True)

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct()(inputs, targets_template=targets_template, forcings=forcings)

    run = jax.jit(lambda rng, inputs, targets_template, forcings: run_forward.apply(
        ckpt.params, {}, rng, inputs, targets_template, forcings)[0]
    )
    return run, rollout


def _gencast_predictor(ckpt):
    import copy
    import haiku as hk
    import jax
    from graphcast import denoiser, gencast, nan_cleaning, normalization, rollout

    stats = _stats(GENCAST_STATS)
    import xarray as xr

    min_by_level = xr.open_dataset(GENCAST_STATS / "min_by_level.nc")
    # DeepMind's official GPU instructions require the GPU-compatible
    # triblockdiag implementation; splash attention is TPU-only in JAX.
    denoiser_config = copy.deepcopy(ckpt.denoiser_architecture_config)
    denoiser_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_config.sparse_transformer_config.mask_type = "full"

    def construct():
        predictor = gencast.GenCast(
            sampler_config=ckpt.sampler_config,
            task_config=ckpt.task_config,
            denoiser_architecture_config=denoiser_config,
            noise_config=ckpt.noise_config,
            noise_encoder_config=ckpt.noise_encoder_config,
        )
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"],
        )
        return nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean="sea_surface_temperature",
        )

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct()(inputs, targets_template=targets_template, forcings=forcings)

    run = jax.jit(lambda rng, inputs, targets_template, forcings: run_forward.apply(
        ckpt.params, {}, rng, inputs, targets_template, forcings)[0]
    )
    return run, rollout


def _native_prediction(prediction, variables, lead_hours, native_step):
    import xarray as xr

    arrays = []
    for variable in variables:
        source = prediction[variable]
        if "sample" in source.dims:
            source = source.mean("sample")
        selected = []
        for lead in lead_hours:
            if lead % native_step:
                raise ValueError(f"Lead {lead}h is not a multiple of official step {native_step}h")
            selected.append(source.sel(time=np.timedelta64(int(lead), "h")))
        da = xr.concat(selected, dim="lead_time").assign_coords(lead_time=np.asarray(lead_hours, dtype=np.int32))
        if "level" in da.dims:
            da = da.transpose("lead_time", "level", "lat", "lon")
        else:
            da = da.transpose("lead_time", "lat", "lon")
        arrays.append(da)
    return arrays


def _native_array(arrays, variables, levels):
    import xarray as xr

    stacked = []
    for variable, da in zip(variables, arrays):
        if "level" in da.dims:
            da = da.sel(level=levels).assign_coords(variable=variable)
            da = da.rename({"level": "level_out"})
            for level in levels:
                item = da.sel(level_out=level).drop_vars("level_out")
                item = item.assign_coords(variable=f"{variable}{int(level)}")
                stacked.append(item)
        else:
            item = da.assign_coords(variable=variable)
            stacked.append(item)
    return stacked


def _write_one(ds, out_path, first):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out_path, mode="w" if first else "a", append_dim=None if first else "init_time", consolidated=False)


def _build_graph_or_gen(cfg, output_path, model_name):
    import jax
    import xarray as xr
    from graphcast import checkpoint, graphcast, gencast

    if model_name == "graphcast":
        checkpoint_path, checkpoint_type = GRAPHCAST_CHECKPOINT, graphcast.CheckPoint
        stats_root, native_step = GRAPHCAST_STATS, 6
    else:
        checkpoint_path, checkpoint_type = GENCAST_CHECKPOINT, gencast.CheckPoint
        stats_root, native_step = GENCAST_STATS, 12
    with checkpoint_path.open("rb") as handle:
        ckpt = checkpoint.load(handle, checkpoint_type)
    run, rollout = _graphcast_predictor(ckpt) if model_name == "graphcast" else _gencast_predictor(ckpt)
    truth, input_data = _open_truth_and_input(cfg)
    if model_name == "gencast":
        # The official GenCast spherical-noise sampler requires monotonically
        # increasing latitude coordinates.  GraphCast itself accepts either
        # orientation; this branch keeps the model-specific reference contract.
        input_data = input_data.sortby("lat", ascending=True)
    variables = _variables()
    leads = [int(x) for x in cfg["experiment"]["lead_time_hours"]]
    daily_samples = daily_mean_sample_hours(leads)
    sample_hours = [hour for lead in leads for hour in daily_samples[lead]]
    max_lead = max(sample_hours)
    init_times = _init_times(cfg)
    out_path = Path(output_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    first = True
    started = time.perf_counter()
    for index, init_time in enumerate(init_times):
        print(f"{model_name} init {index + 1}/{len(init_times)} {pd.to_datetime(init_time)}: loading official ERA5", flush=True)
        inputs, targets, forcings = _model_input_dataset(input_data, init_time, max_lead, ckpt.task_config)
        if model_name == "graphcast":
            prediction = rollout.chunked_prediction(
                run,
                rng=jax.random.PRNGKey(0),
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
            )
            ensemble_std = None
        else:
            members = int(cfg["experiment"].get("gencast_ensemble_members", 8))
            predictions = []
            rng = jax.random.PRNGKey(0)
            for member in range(members):
                print(f"{model_name} init {index + 1}/{len(init_times)} member {member + 1}/{members}", flush=True)
                one = rollout.chunked_prediction(
                    run,
                    rng=jax.random.fold_in(rng, member),
                    inputs=inputs,
                    targets_template=targets * np.nan,
                    forcings=forcings,
                )
                predictions.append(one.expand_dims(sample=[member]))
            prediction = xr.concat(predictions, dim="sample")
            ensemble_std = None
        lat = np.asarray(input_data.lat.values, dtype=np.float32)
        lon = np.asarray(input_data.lon.values, dtype=np.float32)
        target_lat = np.linspace(90.0, -90.0, 121, dtype=np.float32)
        target_lon = np.linspace(0.0, 360.0 - 1.5, 240, dtype=np.float32)
        model_sources = {
            "z": "geopotential",
            "q": "specific_humidity",
            "t": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "t2m": "2m_temperature",
            "mslp": "mean_sea_level_pressure",
        }
        output_map = {}
        for variable in variables:
            if variable[0] in "zqtuv" and variable[1:].isdigit() and int(variable[1:]) in ckpt.task_config.pressure_levels:
                source = prediction[model_sources[variable[0]]]
                source = source.mean("sample") if "sample" in source.dims else source
                source = source.isel(batch=0, drop=True) if "batch" in source.dims else source
                level = int(variable[1:])
                if level in np.asarray(source.level.values).astype(int):
                    item = source.sel(time=np.asarray(sample_hours, dtype="timedelta64[h]"), level=level)
                else:
                    # Some reference-code versions keep pressure levels as
                    # positional indices on the prediction template.
                    level_index = list(ckpt.task_config.pressure_levels).index(level)
                    item = source.sel(
                        time=np.asarray(sample_hours, dtype="timedelta64[h]"),
                        level=source.level.values[level_index],
                    )
            elif variable in {"u10", "v10", "t2m", "mslp"}:
                source = prediction[model_sources[variable]]
                source = source.mean("sample") if "sample" in source.dims else source
                source = source.isel(batch=0, drop=True) if "batch" in source.dims else source
                item = source.sel(time=np.asarray(sample_hours, dtype="timedelta64[h]"))
            else:
                # The protocol has no additional surface field.
                raise RuntimeError(f"Unexpected protocol variable {variable}")
            item = item.assign_coords(lat=lat, lon=lon)
            item = item.interp(lat=target_lat, lon=target_lon, method="linear")
            sampled = np.asarray(item.values, dtype=np.float32)
            output_map[variable] = sampled.reshape(
                len(leads), 4, len(target_lat), len(target_lon)
            ).mean(axis=1)
        pred_values = np.stack([output_map[v] for v in variables], axis=1).astype(np.float32)
        ds = xr.Dataset(
            {
                "pred": (("init_time", "lead_time", "variable", "lat", "lon"), pred_values[None]),
                "valid_time": (("init_time", "lead_time"), np.asarray([[init_time + np.timedelta64(h, "h") for h in leads]], dtype="datetime64[ns]")),
            },
            coords={"init_time": [init_time], "lead_time": leads, "variable": variables,
                    "lat": target_lat, "lon": target_lon},
            attrs={
                "model_name": model_name,
                "source_type": "official_inference",
                "source_uri": GRAPHCAST_SOURCE_URI if model_name == "graphcast" else GENCAST_SOURCE_URI,
                "checkpoint": str(checkpoint_path),
                "initial_condition_uri": WB13_0P25_URI,
                "target_source": cfg["weatherbench2"]["era5_truth_uri"],
                "protocol_variables": json.dumps(variables),
                "allow_synthetic": "false",
                "grid_resolution": "121x240",
                "native_resolution": "181x360",
                "official_native_step_hours": str(native_step),
                "output_temporal_statistic": (
                    "daily mean of forecast states at 00,06,12,18 UTC on each target day"
                ),
                "embedded_target": "omitted; canonical daily S2S truth is loaded during evaluation",
                "ensemble_members": "1" if model_name == "graphcast" else str(cfg["experiment"].get("gencast_ensemble_members", 8)),
                "device_info": str(get_device_info()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_wall_seconds": f"{time.perf_counter() - started:.6f}",
            },
        )
        _write_one(ds, out_path, first)
        first = False
        print(f"{model_name} wrote init {index + 1}/{len(init_times)}", flush=True)
        del inputs, targets, forcings, prediction, ds
    import zarr

    zarr.consolidate_metadata(str(out_path))
    return xr.open_zarr(out_path, consolidated=True)


def build_graphcast_cache(cfg: dict, output_path: str):
    return _build_graph_or_gen(cfg, output_path, "graphcast")


def build_gencast_cache(cfg: dict, output_path: str):
    return _build_graph_or_gen(cfg, output_path, "gencast")
