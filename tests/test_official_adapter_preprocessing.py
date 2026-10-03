import numpy as np

from src.data_sources.fuxi_official import _relative_humidity_fraction_to_percent
from src.data_sources.oneforecast_official import (
    _from_oneforecast_longitude_order,
    _to_oneforecast_longitude_order,
)


def test_fuxi_relative_humidity_fraction_is_converted_to_percent():
    source = np.asarray([-0.1, 0.0, 0.25, 1.0, 1.1], dtype=np.float32)
    actual = _relative_humidity_fraction_to_percent(source)
    np.testing.assert_allclose(actual, [0.0, 0.0, 25.0, 100.0, 100.0])


def test_oneforecast_longitude_conversion_matches_official_h5_order_and_roundtrips():
    source = np.arange(2 * 3 * 240, dtype=np.float32).reshape(2, 3, 240)
    native = _to_oneforecast_longitude_order(source)
    np.testing.assert_array_equal(native[..., 0], source[..., 120])
    np.testing.assert_array_equal(native[..., 120], source[..., 0])
    np.testing.assert_array_equal(_from_oneforecast_longitude_order(native), source)
