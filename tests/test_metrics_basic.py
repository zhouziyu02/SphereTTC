import torch

from src.ttc.metrics import compute_metrics


def test_metrics_zero_error():
    pred = torch.ones((2, 1, 1, 3, 4))
    target = torch.ones((2, 1, 1, 3, 4))
    metrics = compute_metrics(pred, target, [90, 0, -90], torch.device("cpu"))
    assert float(metrics["rmse"][0, 0]) == 0.0
    assert float(metrics["mae"][0, 0]) == 0.0
    assert float(metrics["bias"][0, 0]) == 0.0
