import numpy as np

from src.utils.time import assert_valid_time


REQUIRED_VARS = {"pred", "target", "valid_time"}
REQUIRED_COORDS = {"init_time", "lead_time", "variable", "lat", "lon"}
REQUIRED_ATTRS = {"model_name", "source_type", "source_uri", "device_info"}


def validate_cache_dataset(ds):
    missing_vars = REQUIRED_VARS - set(ds.data_vars)
    missing_coords = REQUIRED_COORDS - set(ds.coords)
    missing_attrs = REQUIRED_ATTRS - set(ds.attrs)
    if missing_vars or missing_coords or missing_attrs:
        raise AssertionError(
            f"Cache schema missing vars={sorted(missing_vars)} coords={sorted(missing_coords)} attrs={sorted(missing_attrs)}"
        )
    if tuple(ds["pred"].shape) != tuple(ds["target"].shape):
        raise AssertionError("pred and target must have identical shape.")
    if ds["pred"].dims != ("init_time", "lead_time", "variable", "lat", "lon"):
        raise AssertionError("pred dims must be init_time, lead_time, variable, lat, lon.")
    lat = np.asarray(ds["lat"].values)
    if lat.size > 1 and not np.all(np.diff(lat) < 0):
        raise AssertionError("Cache latitude must be descending.")
    assert_valid_time(ds["init_time"].values, ds["lead_time"].values, ds["valid_time"].values)
    finite = np.isfinite(ds["pred"].values).mean()
    if finite < 0.95:
        raise AssertionError("pred finite ratio below 95%.")
