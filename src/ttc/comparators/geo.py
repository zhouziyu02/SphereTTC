from __future__ import annotations

import torch


def apply_geo_ttc(
    pred: torch.Tensor,
    pred_memory: torch.Tensor,
    target_memory: torch.Tensor,
    n_modes: int,
    snr_lambda: float,
    strength: float,
    min_memory: int,
    sample_tau: float,
) -> torch.Tensor:
    """Reliability-gated low-zonal-wave affine calibration.

    pred has shape (..., variable, lat, lon) or (variable, lat, lon).
    pred_memory and target_memory have shape (memory, variable, lat, lon).
    """
    if pred_memory.shape[0] < int(min_memory):
        return pred

    x = pred.float()
    finite = torch.isfinite(x)
    x_safe = torch.nan_to_num(x)
    x_mem = torch.nan_to_num(pred_memory.float())
    y_mem = torch.nan_to_num(target_memory.float())
    x_coeff = torch.fft.rfft(x_mem, dim=-1)
    y_coeff = torch.fft.rfft(y_mem, dim=-1)
    k = min(int(n_modes), x_coeff.shape[-1])
    x_low = x_coeff[..., :k]
    y_low = y_coeff[..., :k]
    mu_x = x_low.mean(dim=0)
    mu_y = y_low.mean(dim=0)
    x_centered = x_low - mu_x.unsqueeze(0)
    y_centered = y_low - mu_y.unsqueeze(0)
    var_x = x_centered.abs().square().mean(dim=0)
    cov_yx = (y_centered * x_centered.conj()).mean(dim=0)
    ridge = 1e-4 * var_x.detach().mean().clamp_min(1e-12)
    scale = cov_yx / (var_x + ridge)
    bias = mu_y - scale * mu_x

    raw_mse = (y_low - x_low).abs().square().mean(dim=0)
    fit = scale.unsqueeze(0) * x_low + bias.unsqueeze(0)
    fit_mse = (y_low - fit).abs().square().mean(dim=0)
    gain = (raw_mse - fit_mse).clamp_min(0.0)
    snr_gate = gain / (gain + float(snr_lambda) * fit_mse / float(max(1, x_low.shape[0])) + 1e-12)
    sample_gate = float(x_low.shape[0]) / (float(x_low.shape[0]) + float(sample_tau))
    gate = snr_gate.clamp(0.0, 1.0) * sample_gate * float(strength)

    current_ft = torch.fft.rfft(x_safe, dim=-1)
    calibrated_ft = current_ft.clone()
    current_low = current_ft[..., :k]
    affine_low = scale * current_low + bias
    calibrated_ft[..., :k] = current_low + gate.to(current_ft.dtype) * (affine_low - current_low)
    calibrated = torch.fft.irfft(calibrated_ft, n=x.shape[-1], dim=-1)
    return torch.where(finite, calibrated, x)
