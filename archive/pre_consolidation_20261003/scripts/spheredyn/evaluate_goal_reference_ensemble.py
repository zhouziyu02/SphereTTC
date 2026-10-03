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


BACKBONES = (
    "spheredyn",
    "convlstm",
    "transformer",
    "fno",
    "vit",
    "cirt",
    "climode",
)
REFERENCES = (
    "fourcastnetv2",
    "oneforecast",
    "fuxi",
    "pangu",
    "graphcast",
)
RIDGE_RATIOS = np.asarray(
    [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    dtype=np.float64,
)
SPATIAL_SHRINKAGES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])


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


def _ridge_weights(covariance: np.ndarray, ratio: float) -> np.ndarray:
    features = covariance.shape[0]
    scale = max(float(np.trace(covariance)) / features, 1e-18)
    matrix = 0.5 * (covariance + covariance.T)
    matrix = matrix + (float(ratio) * scale + 1e-12 * scale) * np.eye(features)
    ones = np.ones(features)
    try:
        direction = np.linalg.solve(matrix, ones)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(matrix, rcond=1e-12) @ ones
    denominator = float(ones @ direction)
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.full(features, 1.0 / features)
    weights = direction / denominator
    if not np.all(np.isfinite(weights)):
        return np.full(features, 1.0 / features)
    return weights


def _energy(
    second_moment: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> float:
    return float(
        weights @ second_moment @ weights
        + 2.0 * bias * (weights @ mean)
        + bias * bias
    )


def _candidate_weights(
    covariance: np.ndarray,
    minimum_backbone_weight: float,
) -> list[tuple[str, int, float, np.ndarray]]:
    features = covariance.shape[0]
    if minimum_backbone_weight > 0:
        reference_features = features - 1
        reference_covariance = covariance[:-1, :-1]
        reference_candidates: list[tuple[str, int, np.ndarray]] = [
            ("anchor_ridge", index, _ridge_weights(reference_covariance, ratio))
            for index, ratio in enumerate(RIDGE_RATIOS)
        ]
        reference_candidates.append(
            ("anchor_uniform", -1, np.full(reference_features, 1.0 / reference_features))
        )
        for index in range(reference_features):
            weights = np.zeros(reference_features)
            weights[index] = 1.0
            reference_candidates.append(("anchor_onehot", index, weights))
        anchors = sorted(
            {
                float(minimum_backbone_weight),
                max(float(minimum_backbone_weight), 0.5),
                max(float(minimum_backbone_weight), 0.75),
            }
        )
        candidates = []
        for kind, index, reference_weights in reference_candidates:
            for anchor in anchors:
                if anchor >= 1.0:
                    continue
                weights = np.zeros(features)
                weights[:-1] = (1.0 - anchor) * reference_weights
                weights[-1] = anchor
                candidates.append((kind, index, anchor, weights))
        raw = np.zeros(features)
        raw[-1] = 1.0
        candidates.append(("raw_noop", -1, 1.0, raw))
        return candidates
    candidates = [
        ("ridge", index, float("nan"), _ridge_weights(covariance, ratio))
        for index, ratio in enumerate(RIDGE_RATIOS)
    ]
    candidates.append(
        ("uniform", -1, float("nan"), np.full(features, 1.0 / features))
    )
    for index in range(features):
        weights = np.zeros(features)
        weights[index] = 1.0
        candidates.append(("one_hot", index, float("nan"), weights))
    return candidates


def _feature_statistics(
    statistics: dict[str, np.ndarray],
    backbone_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    reference_count = statistics["reference_sum"].shape[-1]
    cross = np.zeros(
        statistics["reference_cross"].shape[:2]
        + (reference_count + 1, reference_count + 1),
        dtype=np.float64,
    )
    cross[..., :reference_count, :reference_count] = statistics[
        "reference_cross"
    ]
    cross[..., :reference_count, reference_count] = statistics[
        "raw_reference_cross"
    ][backbone_index]
    cross[..., reference_count, :reference_count] = statistics[
        "raw_reference_cross"
    ][backbone_index]
    cross[..., reference_count, reference_count] = statistics["raw_sse"][
        backbone_index
    ]
    sums = np.concatenate(
        [
            statistics["reference_sum"],
            statistics["raw_sum"][backbone_index, ..., None],
        ],
        axis=-1,
    )
    return cross, sums, float(statistics["count"])


def _fit_backbone(
    backbone_index: int,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    latitude: np.ndarray,
    minimum_backbone_weight: float,
) -> dict[str, np.ndarray]:
    train_cross, train_sum, train_count = _feature_statistics(train, backbone_index)
    val_cross, val_sum, val_count = _feature_statistics(validation, backbone_index)
    full_cross = train_cross + val_cross
    full_sum = train_sum + val_sum
    full_count = train_count + val_count
    leads, variables, features = train_sum.shape

    selected_kind = np.empty((leads, variables), dtype="U16")
    selected_index = np.zeros((leads, variables), dtype=np.int16)
    selected_anchor = np.full((leads, variables), np.nan, dtype=np.float64)
    train_weights = np.zeros((leads, variables, features), dtype=np.float64)
    train_bias = np.zeros((leads, variables), dtype=np.float64)

    for lead in range(leads):
        for variable in range(variables):
            train_second = train_cross[lead, variable] / train_count
            train_mean = train_sum[lead, variable] / train_count
            train_covariance = train_second - np.outer(train_mean, train_mean)
            val_second = val_cross[lead, variable] / val_count
            val_mean = val_sum[lead, variable] / val_count
            best: tuple[float, str, int, float, np.ndarray, float] | None = None
            for kind, index, anchor, weights in _candidate_weights(
                train_covariance, minimum_backbone_weight
            ):
                bias = -float(weights @ train_mean)
                score = _energy(val_second, val_mean, weights, bias)
                candidate = (score, kind, index, anchor, weights, bias)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            assert best is not None
            _, kind, index, anchor, weights, bias = best
            selected_kind[lead, variable] = kind
            selected_index[lead, variable] = index
            selected_anchor[lead, variable] = anchor
            train_weights[lead, variable] = weights
            train_bias[lead, variable] = bias

    reference_count = train["reference_map_sum"].shape[-1]
    train_feature_map = np.concatenate(
        [
            train["reference_map_sum"],
            train["raw_map_sum"][backbone_index, ..., None],
        ],
        axis=-1,
    )
    val_feature_map = np.concatenate(
        [
            validation["reference_map_sum"],
            validation["raw_map_sum"][backbone_index, ..., None],
        ],
        axis=-1,
    )
    train_mean_map = (
        np.einsum("lvxym,lvm->lvxy", train_feature_map, train_weights, optimize=True)
        / float(train["days"])
        + train_bias[..., None, None]
    )
    val_sum_map = (
        np.einsum("lvxym,lvm->lvxy", val_feature_map, train_weights, optimize=True)
        + float(validation["days"]) * train_bias[..., None, None]
    )
    spatial_shrink = np.zeros((leads, variables), dtype=np.float64)
    latitude_map = latitude[None, None, :, None]
    linear = np.sum(latitude_map * train_mean_map * val_sum_map, axis=(2, 3))
    quadratic = float(validation["days"]) * np.sum(
        latitude_map * train_mean_map * train_mean_map,
        axis=(2, 3),
    )
    deltas = np.stack(
        [
            -2.0 * value * linear + value * value * quadratic
            for value in SPATIAL_SHRINKAGES
        ],
        axis=0,
    )
    spatial_shrink = SPATIAL_SHRINKAGES[np.argmin(deltas, axis=0)]

    final_weights = np.zeros_like(train_weights)
    final_bias = np.zeros_like(train_bias)
    for lead in range(leads):
        for variable in range(variables):
            full_second = full_cross[lead, variable] / full_count
            full_mean = full_sum[lead, variable] / full_count
            full_covariance = full_second - np.outer(full_mean, full_mean)
            kind = str(selected_kind[lead, variable])
            index = int(selected_index[lead, variable])
            anchor = float(selected_anchor[lead, variable])
            if kind == "anchor_ridge":
                reference_weights = _ridge_weights(
                    full_covariance[:-1, :-1], RIDGE_RATIOS[index]
                )
                weights = np.zeros(features)
                weights[:-1] = (1.0 - anchor) * reference_weights
                weights[-1] = anchor
            elif kind == "anchor_onehot":
                weights = np.zeros(features)
                weights[index] = 1.0 - anchor
                weights[-1] = anchor
            elif kind == "anchor_uniform":
                weights = np.zeros(features)
                weights[:-1] = (1.0 - anchor) / (features - 1)
                weights[-1] = anchor
            elif kind == "raw_noop":
                weights = np.zeros(features)
                weights[-1] = 1.0
            elif kind == "ridge":
                weights = _ridge_weights(full_covariance, RIDGE_RATIOS[index])
            elif kind == "one_hot":
                weights = np.zeros(features)
                weights[index] = 1.0
            else:
                weights = np.full(features, 1.0 / features)
            final_weights[lead, variable] = weights
            final_bias[lead, variable] = -float(weights @ full_mean)

    full_feature_map = train_feature_map + val_feature_map
    full_mean_map = (
        np.einsum("lvxym,lvm->lvxy", full_feature_map, final_weights, optimize=True)
        / float(train["days"] + validation["days"])
        + final_bias[..., None, None]
    )
    spatial_bias = spatial_shrink[..., None, None] * full_mean_map
    return {
        "weights": final_weights,
        "bias": final_bias,
        "spatial_bias": spatial_bias,
        "spatial_shrink": spatial_shrink,
        "selected_kind": selected_kind,
        "selected_index": selected_index,
        "selected_anchor": selected_anchor,
        "reference_count": np.asarray(reference_count),
    }

