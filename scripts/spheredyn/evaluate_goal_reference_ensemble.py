#!/usr/bin/env python
"""Fit a 2017-only reference-ensemble SphereTTC and score 2018--2019.

The same affine stacking algorithm is applied to each of the seven paper
backbones.  Five released-system forecasts provide fixed reference experts;
the sixth expert is the backbone being corrected.  All weights, global bias,
and protected spatial-bias shrinkage are selected using 2017 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import xarray as xr

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from src.experiments.protocol import COMMON_EVALUATION_VARIABLES

# The SphereTTC-v22 (Mode B) math lives in src/spherettc.py; it is re-exported
# here for scripts/spheredyn/run_main_spherettc_v2.py.
from src.spherettc import (  # noqa: F401
    REFERENCES,
    RIDGE_RATIOS,
    SPATIAL_SHRINKAGES,
    _candidate_weights,
    _energy,
    _feature_statistics,
    _fit_backbone,
    _ridge_weights,
)


BACKBONES = (
    "spheredyn",
    "convlstm",
    "transformer",
    "fno",
    "vit",
    "cirt",
    "climode",
)


def _open(path: Path, variables: list[str]) -> xr.Dataset:
    dataset = xr.open_zarr(path, consolidated=True).sel(variable=variables)
    return dataset.transpose(
        "init_time", "lead_time", "variable", "lat", "lon"
    )


def _coordinates(dataset: xr.Dataset) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(dataset[name].values)
        for name in ("init_time", "lead_time", "variable", "lat", "lon")
    )


def _validate_coordinates(reference: xr.Dataset, other: xr.Dataset, name: str) -> None:
    for coordinate, left, right in zip(
        ("init_time", "lead_time", "variable", "lat", "lon"),
        _coordinates(reference),
        _coordinates(other),
        strict=True,
    ):
        if not np.array_equal(left, right):
            raise RuntimeError(f"{name}: {coordinate} coordinate mismatch")


def _prediction(dataset: xr.Dataset, indices: slice) -> np.ndarray:
    return np.asarray(dataset["pred"].isel(init_time=indices).values, dtype=np.float32)


def _target(dataset: xr.Dataset, indices: slice) -> np.ndarray:
    return np.asarray(dataset["target"].isel(init_time=indices).values, dtype=np.float32)


def _weighted_sse(error: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    return np.einsum("blvxy,x->lv", error * error, latitude, optimize=True)


def _weighted_daily_sse(error: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    return np.einsum("blvxy,x->blv", error * error, latitude, optimize=True)


def _block_indices(count: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = min(int(block), int(count))
    starts = rng.integers(0, count - block + 1, int(np.ceil(count / block)))
    return np.concatenate(
        [np.arange(start, start + block) for start in starts]
    )[:count]

