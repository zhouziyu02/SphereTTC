import numpy as np

from src.utils.climatology import calendar_day_index


def test_calendar_day_index_preserves_month_day_across_leap_and_nonleap_years():
    values = np.asarray(
        [
            "2019-02-28",
            "2019-03-01",
            "2020-02-28",
            "2020-02-29",
            "2020-03-01",
            "2020-12-31",
        ],
        dtype="datetime64[D]",
    )
    np.testing.assert_array_equal(calendar_day_index(values), [58, 60, 58, 59, 60, 365])
