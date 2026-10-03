import os
from typing import Any, Dict


def get_device_info() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_importable": False,
            "torch_error": repr(exc),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "device": "unavailable",
        }
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    return {
        "torch_importable": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": available,
        "device_count": count,
        "device_names": [torch.cuda.get_device_name(i) for i in range(count)],
        "device": "cuda:0" if available else "cpu",
    }


def require_cuda(step: str = "experiment"):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for {step}; refusing to run formal inference/TTC/evaluation on CPU."
        )
    return torch.device("cuda")


def select_device(require: bool = True):
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if require:
        raise RuntimeError("CUDA unavailable and require_cuda=true.")
    return torch.device("cpu")
