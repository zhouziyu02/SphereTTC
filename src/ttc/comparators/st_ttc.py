from __future__ import annotations

import torch
from torch import nn


class STTTCSDCalibrator(nn.Module):
    """Official ST-TTC-style horizon spectral domain calibrator.

    This is the weather-cache adapter of the NeurIPS 2025 SDCalibrator: FFT is
    taken along the forecast-horizon axis, and per-node amplitude/phase offsets
    are updated with delayed labels while the backbone remains frozen.
    """

    def __init__(self, n_leads: int, n_nodes: int, groups: int = 3):
        super().__init__()
        self.n_leads = int(n_leads)
        self.freq_bins = self.n_leads // 2 + 1
        self.groups = min(max(1, int(groups)), self.freq_bins)
        self.group_size = max(1, self.freq_bins // self.groups)
        self.lambda_amp = nn.Parameter(torch.zeros(self.groups, n_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(self.groups, n_nodes, 1))

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        """Calibrate a tensor shaped (batch, lead, variable, lat, lon)."""
        batch, leads, variables, nlat, nlon = prediction.shape
        if leads != self.n_leads:
            raise ValueError(f"expected {self.n_leads} leads, got {leads}")
        nodes = variables * nlat * nlon
        values = prediction.permute(0, 2, 3, 4, 1).reshape(batch, nodes, leads)
        spectrum = torch.fft.rfft(values.float(), dim=-1)
        amplitude = spectrum.abs()
        phase = torch.angle(spectrum)
        corrected = torch.empty_like(spectrum)
        for group in range(self.groups):
            start = group * self.group_size
            end = self.freq_bins if group == self.groups - 1 else min(
                self.freq_bins, (group + 1) * self.group_size
            )
            amp = amplitude[..., start:end] * (1.0 + self.lambda_amp[group].unsqueeze(0))
            phi = phase[..., start:end] + self.lambda_phi[group].unsqueeze(0)
            corrected[..., start:end] = amp * torch.exp(1j * phi)
        restored = torch.fft.irfft(corrected, n=leads, dim=-1)
        return restored.reshape(batch, variables, nlat, nlon, leads).permute(0, 4, 1, 2, 3)
