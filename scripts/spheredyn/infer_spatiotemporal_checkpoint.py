#!/usr/bin/env python
"""Run a frozen local forecast checkpoint on an explicit daily date range."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import dask.array as da
import numpy as np
import torch
import xarray as xr
import zarr

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.baselines import build_model
from src.data_sources.s2s_daily import S2SDailyBackend, daily_range
from src.utils.climatology import calendar_day_index
from src.utils.device import get_device_info, require_cuda
from src.utils.time import make_valid_time


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seasonal_context(
    init_time: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    day = calendar_day_index(init_time).astype(np.float32)
    angle = 2.0 * np.pi * day / 366.0
    values = np.stack(
        [
            np.sin(angle),
            np.cos(angle),
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
        ],
        axis=-1,
    ).astype(np.float32)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _create_cache(
    output: Path,
    init_time: np.ndarray,
    lead_time: np.ndarray,
    variables: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    attrs: dict[str, str],
    include_target: bool = True,
) -> zarr.Group:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (
        len(init_time),
        len(lead_time),
        len(variables),
        len(lat),
        len(lon),
    )
    chunks = (1, 1, shape[2], shape[3], shape[4])
    data_vars = {
        "pred": (
            ("init_time", "lead_time", "variable", "lat", "lon"),
            da.empty(shape, chunks=chunks, dtype="float32"),
        ),
        "valid_time": (
            ("init_time", "lead_time"),
            make_valid_time(init_time, lead_time),
        ),
    }
    if include_target:
        data_vars["target"] = (
            ("init_time", "lead_time", "variable", "lat", "lon"),
            da.empty(shape, chunks=chunks, dtype="float32"),
        )
    dataset = xr.Dataset(
        data_vars,
        coords={
            "init_time": init_time.astype("datetime64[ns]"),
            "lead_time": lead_time.astype("int32"),
            "variable": variables,
            "lat": lat,
            "lon": lon,
        },
        attrs=attrs,
    )
    dataset.to_zarr(output, mode="w", consolidated=False, compute=False)
    return zarr.open_group(str(output), mode="a")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/S2S")
    parser.add_argument("--stats", required=True)
    parser.add_argument(
        "--daily-cache",
        default=None,
        help="Optional frozen contiguous daily cache used to reduce source-data I/O.",
    )
    parser.add_argument("--init-start", required=True)
    parser.add_argument("--init-end-exclusive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument(
        "--omit-target",
        action="store_true",
        help="Store predictions only; metrics must load canonical truth.",
    )
    args = parser.parse_args()

    device = require_cuda("frozen spatiotemporal checkpoint inference")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = _checkpoint_sha256(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    backend = S2SDailyBackend(
        args.data,
        args.stats,
        io_workers=args.io_workers,
        contiguous_cache=args.daily_cache,
    )
    init_time = daily_range(args.init_start, args.init_end_exclusive)
    if init_time.size == 0:
        raise ValueError("explicit inference range is empty")
    last_target = init_time[-1] + np.timedelta64(
        int(backend.lead_time.max() // 24),
        "D",
    )
    if checkpoint["variables"] != backend.variables:
        raise RuntimeError("checkpoint/backend variable mismatch")
    if not np.array_equal(
        np.asarray(checkpoint["lead_time"], dtype=np.int32),
        backend.lead_time,
    ):
        raise RuntimeError("checkpoint/backend lead-time mismatch")
    if not np.allclose(
        np.asarray(checkpoint["mean"], dtype=np.float32),
        backend.mean,
        rtol=0.0,
        atol=0.0,
    ) or not np.allclose(
        np.asarray(checkpoint["std"], dtype=np.float32),
        backend.std,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("checkpoint/backend normalization mismatch")

    model_name = str(checkpoint["model"])
    model = build_model(
        model_name,
        len(backend.variables),
        len(backend.variables),
        len(backend.lead_time),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    uses_seasonal_context = bool(
        getattr(model, "requires_seasonal_context", False)
    )
    history_steps = int(getattr(model, "history_steps", 1))
    mean = torch.as_tensor(
        backend.mean,
        dtype=torch.float32,
        device=device,
    ).view(*((1, 1, -1, 1, 1) if history_steps > 1 else (1, -1, 1, 1)))
    std = torch.as_tensor(
        backend.std,
        dtype=torch.float32,
        device=device,
    ).view(*((1, 1, -1, 1, 1) if history_steps > 1 else (1, -1, 1, 1)))
    output = Path(args.output)
    group = _create_cache(
        output,
        init_time,
        backend.lead_time,
        backend.variables,
        backend.lat,
        backend.lon,
        {
            "model_name": model_name,
            "source_type": "frozen_checkpoint_inference",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device_info": str(get_device_info()),
            "initialization_range": (
                f"[{args.init_start},{args.init_end_exclusive})"
            ),
        },
        include_target=not args.omit_target,
    )
    with torch.no_grad():
        for start in range(0, len(init_time), args.batch_size):
            stop = min(start + args.batch_size, len(init_time))
            if args.omit_target:
                x_values = (
                    backend.load_history_inputs(
                        init_time[start:stop],
                        history_steps=history_steps,
                    )
                    if history_steps > 1
                    else backend.load_inputs(init_time[start:stop])
                )
                y_values = None
            else:
                x_values, y_values = (
                    backend.load_batch_with_history(
                        init_time[start:stop],
                        history_steps=history_steps,
                    )
                    if history_steps > 1
                    else backend.load_batch(init_time[start:stop])
                )
            x = torch.as_tensor(
                np.array(x_values, copy=True),
                dtype=torch.float32,
                device=device,
            )
            normalized = (x - mean) / std
            model_output = (
                model(
                    normalized,
                    seasonal_context=_seasonal_context(
                        init_time[start:stop],
                        device,
                    ),
                )
                if uses_seasonal_context
                else model(normalized)
            )
            output_mean = mean[:, -1] if mean.ndim == 5 else mean
            output_std = std[:, -1] if std.ndim == 5 else std
            prediction = (
                model_output * output_std[:, None] + output_mean[:, None]
            ).cpu().numpy().astype("float32")
            group["pred"][start:stop] = prediction
            if not args.omit_target:
                group["target"][start:stop] = np.asarray(
                    y_values,
                    dtype=np.float32,
                )
            if stop % 64 == 0 or stop == len(init_time):
                print(f"{model_name} wrote inference {stop}/{len(init_time)}", flush=True)
    zarr.consolidate_metadata(str(output))
    torch.cuda.synchronize(device)
    profile = {
        "stage": "frozen_checkpoint_inference",
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output": str(output),
        "init_start": args.init_start,
        "init_end_exclusive": args.init_end_exclusive,
        "n_initializations": int(len(init_time)),
        "last_valid_time": str(last_target),
        "output_includes_target": not args.omit_target,
        "daily_cache": args.daily_cache,
        "wall_seconds": time.perf_counter() - started,
        "cuda_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "device_info": get_device_info(),
    }
    profile_path = Path(args.profile_output)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(profile, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {output}")
    print(f"PROFILE {profile_path}")


if __name__ == "__main__":
    main()
