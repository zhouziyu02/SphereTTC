"""Resource-bounded adapters for the requested external local baselines.

The official CirT implementation is a 1.5-degree S2S model with a fixed
two-step head.  The adapter keeps its geometry-inspired transformer core but
changes the head to the SOON supervised-cache channel/lead contract.

The official ClimODE implementation is hard-coded around five channels and a
5.625-degree dataset.  The adapter keeps the neural-ODE plus advective term,
but parameterizes the field for the local cache's arbitrary channel count and
grid.  These are therefore unified-protocol exploratory adapters, not claims
of reproducing the papers' native training recipes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn


def _load_official_cirt_model():
    path = Path(__file__).resolve().parents[2] / "external" / "CirT_official" / "CIRT" / "models" / "CirT.py"
    # timm 0.9 imports wandb at package import time.  The installed wandb
    # build is not compatible with this worker's protobuf runtime, while CirT
    # itself does not use wandb.  Stub only that optional logging module.
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    spec = importlib.util.spec_from_file_location("soon_external_cirt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official CirT source at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


class CirTForecast(nn.Module):
    """CirT geometry-inspired transformer with a variable lead head."""

    def __init__(
        self,
        in_channels: int,
        out_vars: int,
        n_leads: int,
        img_size: tuple[int, int] = (121, 240),
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        decoder_depth: int = 2,
    ):
        super().__init__()
        if in_channels != out_vars:
            raise ValueError("CirT adapter currently requires in_channels == out_vars")
        OfficialCirT = _load_official_cirt_model()
        self.n_leads = n_leads
        self.out_vars = out_vars
        self.img_size = tuple(img_size)
        self.core = OfficialCirT(
            img_size=list(img_size),
            input_size=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
        )
        last = self.core.head[-1]
        self.core.head[-1] = nn.Linear(last.in_features, out_vars * n_leads * img_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"CirT expects {self.img_size}, got {(h, w)}")
        tokens = self.core.forward_encoder(x)
        pred = self.core.head(tokens)
        # Official CirT unpatchifies [B, H, 1, W, C*2].  Generalize the fixed
        # two-step head to the common local n_leads contract.
        pred = pred.reshape(b, h, 1, w, self.out_vars * self.n_leads)
        pred = torch.einsum("nhwpc->nchpw", pred)
        return pred.reshape(b, self.n_leads, self.out_vars, h, w)


class _PeriodicResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Longitude is periodic; latitude uses replicated boundary values.
        y = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
        y = nn.functional.pad(y, (0, 0, 1, 1), mode="replicate")
        return x + self.net(y)[..., 1:-1, 1:-1]


class _ClimODEField(nn.Module):
    def __init__(self, channels: int, hidden: int, lat_size: int, lon_size: int):
        super().__init__()
        self.channels = channels
        self.register_buffer("lat", torch.linspace(1.0, -1.0, lat_size).view(1, 1, lat_size, 1))
        self.register_buffer("lon", torch.linspace(-1.0, 1.0, lon_size).view(1, 1, 1, lon_size))
        # state = [velocity_x, velocity_y, field].
        feature_channels = 5 * channels + 4
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, 3, padding=1),
            nn.GELU(),
            _PeriodicResidualBlock(hidden, hidden),
            _PeriodicResidualBlock(hidden, hidden),
            nn.Conv2d(hidden, 2 * channels, 1),
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        c = self.channels
        vx, vy, field = state[:, :c], state[:, c : 2 * c], state[:, 2 * c :]
        grad_y = torch.gradient(field, dim=-2)[0]
        grad_x = torch.gradient(field, dim=-1)[0]
        phase = t.to(field).reshape(1, 1, 1, 1)
        spatial = field[:, :1]
        time_features = torch.cat(
            [torch.sin(phase).expand_as(spatial), torch.cos(phase).expand_as(spatial), self.lat.expand_as(spatial), self.lon.expand_as(spatial)],
            dim=1,
        )
        features = torch.cat([vx, vy, field, grad_x, grad_y, time_features], dim=1)
        learned_vx, learned_vy = self.net(features).split(c, dim=1)
        advective = vx * grad_x + vy * grad_y
        dfield = advective + 0.1 * (learned_vx + learned_vy)
        # Relax velocity toward a learned flow field while evolving the field.
        return torch.cat([learned_vx - vx, learned_vy - vy, dfield], dim=1)


class ClimODEForecast(nn.Module):
    """Channel- and grid-parameterized ClimODE-style neural ODE rollout."""

    def __init__(
        self,
        in_channels: int,
        out_vars: int,
        n_leads: int,
        img_size: tuple[int, int] = (121, 240),
        hidden: int = 64,
        solver: str = "euler",
    ):
        super().__init__()
        if in_channels != out_vars:
            raise ValueError("ClimODE adapter currently requires in_channels == out_vars")
        if solver != "euler":
            raise ValueError("The local ClimODE adapter currently supports solver='euler' only")
        self.n_leads = n_leads
        self.field = _ClimODEField(in_channels, hidden, img_size[0], img_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        state = torch.cat([torch.zeros_like(x), torch.zeros_like(x), x], dim=1)
        # One ODE interval corresponds to one requested lead.  The output at
        # t=1..n_leads is the same [B, lead, variable, lat, lon] contract used
        # by the existing local baselines.
        outputs = []
        for lead in range(self.n_leads):
            state = state + self.field(torch.tensor(float(lead), device=x.device), state)
            outputs.append(state[:, 2 * c :].contiguous())
        return torch.stack(outputs, dim=1)
