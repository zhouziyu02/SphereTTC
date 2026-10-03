import numpy as np
import xarray as xr

from src.utils.cache_schema import validate_cache_dataset


def test_cache_schema_validates_minimal_dataset():
    init = np.array(["2020-01-01T00"], dtype="datetime64[h]")
    lead = np.array([24], dtype=np.int32)
    data = np.ones((1, 1, 1, 2, 2), dtype=np.float32)
    ds = xr.Dataset(
        {
            "pred": (("init_time", "lead_time", "variable", "lat", "lon"), data),
            "target": (("init_time", "lead_time", "variable", "lat", "lon"), data),
            "valid_time": (("init_time", "lead_time"), init[:, None] + lead.astype("timedelta64[h]")[None, :]),
        },
        coords={"init_time": init, "lead_time": lead, "variable": ["t2m"], "lat": [90.0, -90.0], "lon": [0.0, 1.5]},
        attrs={"model_name": "unit", "source_type": "unit", "source_uri": "unit://not-real", "device_info": "{}"},
    )
    validate_cache_dataset(ds)
