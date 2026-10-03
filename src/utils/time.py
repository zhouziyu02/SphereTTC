import numpy as np


def as_timedelta64_hours(hours):
    return np.asarray(hours, dtype="timedelta64[h]")


def lead_hours_from_values(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return (arr / np.timedelta64(1, "h")).astype(int)
    return arr.astype(int)


def make_valid_time(init_times, lead_time_hours):
    init = np.asarray(init_times, dtype="datetime64[ns]")[:, None]
    lead = np.asarray(lead_time_hours, dtype="timedelta64[h]")[None, :]
    return init + lead


def assert_valid_time(init_times, lead_time_hours, valid_times):
    expected = make_valid_time(init_times, lead_time_hours)
    actual = np.asarray(valid_times, dtype="datetime64[ns]")
    if actual.shape != expected.shape or not np.all(actual == expected):
        raise AssertionError("valid_time must equal init_time + lead_time for every sample.")
