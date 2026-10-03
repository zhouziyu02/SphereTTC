import numpy as np
import pytest

from src.utils.time import assert_valid_time, make_valid_time


def test_valid_time_alignment():
    init = np.array(["2020-01-01T00", "2020-01-01T12"], dtype="datetime64[h]")
    leads = [24, 72]
    valid = make_valid_time(init, leads)
    assert_valid_time(init, leads, valid)


def test_valid_time_alignment_rejects_bad_value():
    init = np.array(["2020-01-01T00"], dtype="datetime64[h]")
    valid = np.array([["2020-01-03T00"]], dtype="datetime64[h]")
    with pytest.raises(AssertionError):
        assert_valid_time(init, [24], valid)
