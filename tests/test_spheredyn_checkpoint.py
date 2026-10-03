"""The frozen SphereDyn-v9 checkpoint must load strictly into the code in src/."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / (
    "artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/"
    "spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt"
)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="SphereDyn checkpoint not present")
def test_spheredyn_v9_checkpoint_loads_strictly():
    from src.baselines import build_model

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["model"] == "spheredyn_v9_multiscale"
    n_vars, n_leads = len(checkpoint["variables"]), len(checkpoint["lead_time"])
    model = build_model(checkpoint["model"], n_vars, n_vars, n_leads)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 3_966_080
    assert int(getattr(model, "history_steps", 1)) == 3
