import numpy as np
from pathlib import Path
import pytest

from src.data_sources.s2s_daily import load_day, load_targets


if not Path("data/S2S").exists():
    pytest.skip(
        "raw S2S data is intentionally excluded from the migration bundle",
        allow_module_level=True,
    )


def test_daily_loader_variable_order_and_shape():
    values = load_day("data/S2S", "2019-01-01")
    assert values.shape == (54, 121, 240)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()

    # Pressure fields are grouped by source variable, then ascending protocol level.
    import xarray as xr

    pressure = xr.open_zarr(
        "data/S2S/pressure_level_1.5/era5_pressure_full_1.5deg_20190101.zarr",
        consolidated=True,
        mask_and_scale=False,
    )
    surface = xr.open_zarr(
        "data/S2S/single_level_1.5/era5_single_full_1.5deg_20190101.zarr",
        consolidated=True,
        mask_and_scale=False,
    )
    assert np.array_equal(values[0], pressure.geopotential.sel(level=10).values)
    assert np.array_equal(values[30], pressure.u_component_of_wind.sel(level=10).values)
    assert np.array_equal(values[40], pressure.v_component_of_wind.sel(level=10).values)
    assert np.array_equal(values[50], surface["10m_u_component_of_wind"].values)
    assert np.array_equal(values[53], surface["mean_sea_level_pressure"].values)


def test_target_loader_uses_exact_daily_leads_and_variable_subset():
    target = load_targets(
        "data/S2S",
        [np.datetime64("2018-12-21")],
        [24, 240],
        ["z50", "u10", "mslp"],
        workers=2,
    )
    assert target.shape == (1, 2, 3, 121, 240)
    day_1 = load_day("data/S2S", "2018-12-22")
    day_10 = load_day("data/S2S", "2018-12-31")
    assert np.array_equal(target[0, 0], day_1[[1, 50, 53]])
    assert np.array_equal(target[0, 1], day_10[[1, 50, 53]])
