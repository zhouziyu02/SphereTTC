import torch

from src.baselines.spheredyn_v9 import SphereDynV9Forecast


def _model(mode: str) -> SphereDynV9Forecast:
    return SphereDynV9Forecast(
        54,
        54,
        2,
        image_size=(8, 16),
        width=16,
        local_layers=1,
        spectral_rank=4,
        spectral_lmax=6,
        spectral_bands=2,
        observation_width=16,
        observation_layers=1,
        observation_rank=4,
        observation_lmax=6,
        observation_bands=2,
        multiscale_width=16,
        multiscale_layers=1,
        multiscale_mode=mode,
    )


def test_v9_zero_initialized_path_is_exact_v8_noop():
    model = _model("learned").eval()
    history = torch.randn(1, 3, 54, 8, 16)
    seasonal = torch.randn(1, 4)
    with torch.no_grad():
        expected = super(SphereDynV9Forecast, model).forward(history, seasonal)
        actual = model(history, seasonal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v9_control_has_identical_parameters_and_compute_path():
    candidate = _model("learned").eval()
    control = _model("null_control").eval()
    control.load_state_dict(candidate.state_dict())
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters()
    )
    history = torch.randn(1, 3, 54, 8, 16)
    seasonal = torch.randn(1, 4)
    with torch.no_grad():
        torch.testing.assert_close(
            candidate(history, seasonal), control(history, seasonal), rtol=0, atol=0
        )


def test_v9_multiscale_head_receives_finite_gradient():
    model = _model("learned")
    history = torch.randn(1, 3, 54, 8, 16)
    output = model(history, torch.randn(1, 4))
    assert output.shape == (1, 2, 54, 8, 16)
    output.square().mean().backward()
    gradient = model.multiscale_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
