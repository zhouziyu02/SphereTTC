import pytest

from src.utils.device import get_device_info, require_cuda


def test_device_info_has_required_keys():
    info = get_device_info()
    assert "cuda_available" in info
    assert "cuda_visible_devices" in info
    assert "device_count" in info


def test_require_cuda_policy():
    info = get_device_info()
    if not info["cuda_available"]:
        with pytest.raises(RuntimeError):
            require_cuda("unit test")
