"""Small CPU checks for calibration behavior and chronological availability."""

import numpy as np
import pytest
import torch

from spherettc import SphereTTC, SphereTTCCalibrator, SphereTTCConfig


def small_config(**overrides):
    values = dict(
        lmax=4,
        mmax=4,
        min_memory=4,
        min_holdout=2,
        holdout_fraction=0.25,
        confidence_z=0.0,
    )
    values.update(overrides)
    return SphereTTCConfig(**values)


def test_cold_start_is_identity_and_preserves_missing_values():
    calibrator = SphereTTCCalibrator(9, 16, torch.device("cpu"), small_config())
    current = torch.randn(2, 9, 16, requires_grad=True)
    with torch.no_grad():
        current[0, 0, 0] = float("nan")
    memory = calibrator.encode(torch.randn(3, 2, 9, 16))
    output, gate = calibrator.calibrate(current, memory, memory)
    torch.testing.assert_close(output, current, rtol=0, atol=0, equal_nan=True)
    assert torch.count_nonzero(gate) == 0
    assert not output.requires_grad


def test_stable_large_scale_bias_is_corrected():
    torch.manual_seed(7)
    calibrator = SphereTTCCalibrator(
        9, 16, torch.device("cpu"), small_config(ridge=0.01, strength=1.0)
    )
    truth = torch.randn(12, 1, 9, 16)
    bias = torch.linspace(-1.0, 1.0, 9).view(1, 1, 9, 1)
    current_truth = torch.randn(1, 9, 16)
    current_prediction = current_truth + bias[0]
    output, gate = calibrator.calibrate(
        current_prediction,
        calibrator.encode(truth + bias),
        calibrator.encode(truth),
    )
    assert (output - current_truth).square().mean() < (
        current_prediction - current_truth
    ).square().mean()
    assert gate.item() > 0


def test_validation_gate_rejects_a_bias_that_has_disappeared():
    torch.manual_seed(11)
    calibrator = SphereTTCCalibrator(
        9, 16, torch.device("cpu"), small_config(ridge=0.01, confidence_z=1.0)
    )
    truth = torch.randn(12, 1, 9, 16)
    prediction = truth.clone()
    prediction[:9] += 2.0
    current = torch.randn(1, 9, 16)
    output, gate = calibrator.calibrate(
        current, calibrator.encode(prediction), calibrator.encode(truth)
    )
    assert gate.item() == 0
    torch.testing.assert_close(output, current, rtol=0, atol=0)


@pytest.mark.parametrize("delay_hours", [0, 24, 72])
def test_memory_waits_for_each_leads_verification(delay_hours):
    online = SphereTTC(
        9,
        16,
        [24, 72],
        small_config(),
        memory_size=2,
        verification_delay_hours=delay_hours,
    )
    field = torch.zeros(2, 1, 9, 16)
    first = np.datetime64("2018-01-01", "ns")
    for day in range(9):
        now = first + np.timedelta64(day, "D")
        output, gate, counts = online.calibrate(now, field)
        expected = [max(0, min(2, day - lead - delay_hours // 24 + 1)) for lead in [1, 3]]
        assert counts == expected
        assert np.count_nonzero(gate) == 0
        torch.testing.assert_close(output, field, rtol=0, atol=0)
        online.update(now, field, field)


def test_chronological_calls_and_nonnegative_latency_are_required():
    field = np.zeros((1, 1, 9, 16), dtype=np.float32)
    online = SphereTTC(9, 16, [24], small_config())
    online.calibrate("2018-01-02", field)
    with pytest.raises(ValueError, match="strictly increasing"):
        online.calibrate("2018-01-01", field)
    online.update("2018-01-02", field, field)
    with pytest.raises(ValueError, match="strictly increasing"):
        online.update("2018-01-02", field, field)
    with pytest.raises(ValueError, match="nonnegative"):
        SphereTTC(9, 16, [24], verification_delay_hours=-1)
