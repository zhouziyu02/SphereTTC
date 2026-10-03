from __future__ import annotations

import numpy as np


LEAP_TEMPLATE_YEAR = 2000


def calendar_day_index(values) -> np.ndarray:
    """Return zero-based month/day indices on a 366-day leap-year calendar."""
    dates = np.asarray(values).astype("datetime64[D]")
    flat = dates.reshape(-1)
    result = np.empty(flat.size, dtype=np.int16)
    template_start = np.datetime64(f"{LEAP_TEMPLATE_YEAR}-01-01", "D")
    for index, value in enumerate(flat):
        month_day = np.datetime_as_string(value, unit="D")[5:]
        result[index] = int(
            (
                np.datetime64(f"{LEAP_TEMPLATE_YEAR}-{month_day}", "D")
                - template_start
            )
            / np.timedelta64(1, "D")
        )
    return result.reshape(dates.shape)
