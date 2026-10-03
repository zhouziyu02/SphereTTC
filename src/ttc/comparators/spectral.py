import torch


def apply_zonal_spectral(pred, gate, bias, n_modes: int):
    x = pred.float()
    finite = torch.isfinite(x)
    x_safe = torch.nan_to_num(x)
    ft = torch.fft.rfft(x_safe, dim=-1)
    k = min(int(n_modes), ft.shape[-1])
    low = ft[..., :k] * gate[..., :k] + bias[..., :k]
    if k < ft.shape[-1]:
        ft_mod = torch.cat([low, ft[..., k:]], dim=-1)
    else:
        ft_mod = low
    calibrated = torch.fft.irfft(ft_mod, n=x.shape[-1], dim=-1)
    return torch.where(finite, calibrated, x)
