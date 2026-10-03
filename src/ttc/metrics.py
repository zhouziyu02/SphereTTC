from __future__ import annotations

from typing import Dict

import torch


def latitude_weights(lat, device):
    lat_t = torch.as_tensor(lat, dtype=torch.float32, device=device)
    w = torch.cos(torch.deg2rad(lat_t)).clamp_min(0)
    return w / w.mean().clamp_min(1e-12)


def weighted_mean_spatial(x, weights):
    while weights.ndim < x.ndim:
        weights = weights.view(*([1] * (x.ndim - 2)), -1, 1)
    return (x * weights).mean(dim=(-2, -1))


def weighted_rmse(pred, target, weights):
    return torch.sqrt(weighted_mean_spatial((pred - target) ** 2, weights).mean(dim=0))


def weighted_mae(pred, target, weights):
    return weighted_mean_spatial((pred - target).abs(), weights).mean(dim=0)


def weighted_bias(pred, target, weights):
    return weighted_mean_spatial(pred - target, weights).mean(dim=0)


def weighted_acc(pred, target, weights, climatology=None):
    if climatology is None:
        climatology = target.mean(dim=0, keepdim=True)
    pa = pred - climatology
    ta = target - climatology
    num = weighted_mean_spatial(pa * ta, weights).sum(dim=0)
    den = torch.sqrt(weighted_mean_spatial(pa**2, weights).sum(dim=0) * weighted_mean_spatial(ta**2, weights).sum(dim=0))
    return num / den.clamp_min(1e-12)


def spectral_residual(pred, target):
    residual = (pred - target).float()
    ft = torch.fft.rfft(residual, dim=-1)
    return ft.abs().mean(dim=(0, -2, -1))


def compute_metrics(pred, target, lat, device, climatology=None) -> Dict[str, torch.Tensor]:
    pred = pred.to(device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    weights = latitude_weights(lat, device)
    clim = None if climatology is None else climatology.to(device=device, dtype=torch.float32)
    return {
        "rmse": weighted_rmse(pred, target, weights),
        "mae": weighted_mae(pred, target, weights),
        "bias": weighted_bias(pred, target, weights),
        "acc_temporal_mean": weighted_acc(pred, target, weights, clim),
        "spectral_residual": spectral_residual(pred, target),
    }
