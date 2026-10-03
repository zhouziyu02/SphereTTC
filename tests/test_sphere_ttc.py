import torch

from src.spherettc import SphereTTCConfig, SphereTTCCalibrator


def _calibrator(**kwargs):
    values = {
        "lmax": 4,
        "mmax": 4,
        "min_memory": 4,
        "min_holdout": 2,
        "holdout_fraction": 0.25,
        "confidence_z": 0.0,
    }
    values.update(kwargs)
    config = SphereTTCConfig(**values)
    return SphereTTCCalibrator(9, 16, torch.device("cpu"), config)


def test_sphere_ttc_is_identity_without_enough_memory():
    calibrator = _calibrator()
    current = torch.randn(2, 9, 16)
    coeff = calibrator.encode(torch.randn(3, 2, 9, 16))
    output, gate = calibrator.calibrate(current, coeff, coeff)
    assert torch.equal(output, current)
    assert torch.equal(gate, torch.zeros_like(gate))


def test_sphere_ttc_corrects_stable_large_scale_bias():
    torch.manual_seed(7)
    calibrator = _calibrator(ridge=0.01, strength=1.0)
    truth = torch.randn(12, 1, 9, 16)
    latitude_bias = torch.linspace(-1.0, 1.0, 9).view(1, 1, 9, 1)
    prediction = truth + latitude_bias
    pred_coeff = calibrator.encode(prediction)
    target_coeff = calibrator.encode(truth)
    current_truth = torch.randn(1, 9, 16)
    current_prediction = current_truth + latitude_bias[0]
    calibrated, gate = calibrator.calibrate(current_prediction, pred_coeff, target_coeff)
    raw_mse = (current_prediction - current_truth).square().mean()
    calibrated_mse = (calibrated - current_truth).square().mean()
    assert calibrated_mse < raw_mse
    assert gate.item() > 0


def test_sphere_ttc_risk_gate_rejects_unstable_correction():
    torch.manual_seed(11)
    calibrator = _calibrator(ridge=0.01, strength=1.0, confidence_z=1.0)
    truth = torch.randn(12, 1, 9, 16)
    prediction = truth.clone()
    prediction[:9] += 2.0
    pred_coeff = calibrator.encode(prediction)
    target_coeff = calibrator.encode(truth)
    current = torch.randn(1, 9, 16)
    calibrated, gate = calibrator.calibrate(current, pred_coeff, target_coeff)
    assert gate.item() == 0.0
    assert torch.allclose(calibrated, current, atol=1e-5)
