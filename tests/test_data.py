"""Tests for daily sampling, coordinate handling, and fit-only statistics."""
from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from spherettc import data


def source_6h(days: int = 1) -> xr.Dataset:
    times = np.datetime64("2000-01-01", "h") + np.arange(days * 4) * np.timedelta64(6, "h")
    values = np.broadcast_to(np.arange(days * 4, dtype=np.float32)[:, None, None], (days * 4, 121, 240)).copy()
    return xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), values)},
        coords={"time": times.astype("datetime64[ns]"), "latitude": data.LATITUDE[::-1], "longitude": data.LONGITUDE},
    )


def test_calendar_slots_and_variable_mapping():
    dates = np.asarray(["2019-03-01", "2020-02-29", "2019-12-31"], dtype="datetime64[D]")
    np.testing.assert_array_equal(data.calendar_day_of_year(dates), [61, 60, 366])
    assert len(data.canonical_variables()) == 49
    assert len(data.canonical_variables(True)) == 54
    assert data.variable_source("u10") == ("10m_u_component_of_wind", None)
    assert data.variable_source("u10hpa") == ("u_component_of_wind", 10)


def test_four_complete_utc_samples_and_latitude_order():
    source = data.normalize_grid(source_6h())
    result = data.extract_daily_fields(source, "2000-01-01", ("t2m",), "conservative_6h")
    assert result.truth.dims == ("time", "variable", "lat", "lon")
    np.testing.assert_array_equal(result.lat.values, data.LATITUDE)
    np.testing.assert_allclose(result.truth.values, 1.5)
    with pytest.raises(ValueError, match="missing required UTC samples"):
        data.extract_daily_fields(source.isel(time=[0, 1, 3]), "2000-01-01", ("t2m",), "conservative_6h")
    corrupted = source.copy(deep=True)
    corrupted["2m_temperature"].values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        data.extract_daily_fields(corrupted, "2000-01-01", ("t2m",), "conservative_6h")


def test_main_daily_exact_subgrid():
    values = np.arange(721 * 1440, dtype=np.float32).reshape(1, 1, 721, 1440)
    source = xr.Dataset(
        {"geopotential": (("time", "level", "latitude", "longitude"), values)},
        coords={"time": [np.datetime64("2000-01-01", "ns")], "level": [500],
                "latitude": np.linspace(90, -90, 721), "longitude": np.arange(1440) * 0.25},
    )
    result = data.extract_daily_fields(data.normalize_grid(source), "2000-01-01", ("z500",), "main_daily")
    np.testing.assert_array_equal(result.truth.values[0, 0], values[0, 0, ::6, ::6])


def test_daily_store_resume(tmp_path, monkeypatch):
    source = data.normalize_grid(source_6h(2))
    monkeypatch.setattr(data, "open_public_era5", lambda source_kind: source)
    output = tmp_path / "truth.zarr"
    data.write_daily_store(output, "2000-01-01", "2000-01-01", "conservative_6h", ("t2m",))
    data.write_daily_store(output, "2000-01-01", "2000-01-02", "conservative_6h", ("t2m",), resume=True)
    with xr.open_zarr(output, consolidated=True, chunks=None) as loaded:
        np.testing.assert_allclose(loaded.truth.values[:, 0, 0, 0], [1.5, 5.5])
        np.testing.assert_array_equal(loaded.time.values.astype("datetime64[D]"),
                                      np.asarray(["2000-01-01", "2000-01-02"], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="different ERA5 source"):
        data.write_daily_store(output, "2000-01-01", "2000-01-02", "main_daily", ("t2m",), resume=True)


def test_streaming_statistics_ignore_test_dates_and_handle_leap_day(tmp_path):
    dates = data.daily_dates("2000-01-01", "2000-12-31")
    values = np.arange(366 * 2 * 3 * 4, dtype=np.float32).reshape(366, 2, 3, 4) / 100
    stored = np.concatenate([values, np.full((1, 2, 3, 4), 1e6, dtype=np.float32)])
    times = np.concatenate([dates, np.asarray(["2018-01-01"], dtype="datetime64[D]")])
    source = xr.Dataset(
        {"truth": (("time", "variable", "lat", "lon"), stored)},
        coords={"time": times.astype("datetime64[ns]"), "variable": ["t2m", "z500"], "lat": [90, 0, -90], "lon": [0, 90, 180, 270]},
    )
    truth = tmp_path / "truth.zarr"
    source.to_zarr(truth, consolidated=True)
    output = tmp_path / "statistics"
    data.prepare_statistics(truth, output, "2000-01-01", "2000-12-31")
    result = json.loads((output / "stats.json").read_text())
    np.testing.assert_allclose(result["mean"], values.astype(np.float64).mean(axis=(0, 2, 3)), rtol=1e-12)
    np.testing.assert_allclose(result["standard_deviation"], values.astype(np.float64).std(axis=(0, 2, 3)), rtol=1e-12)
    with xr.open_zarr(output / "climatology.zarr", consolidated=True, chunks=None) as climatology:
        np.testing.assert_array_equal(climatology.dayofyear.values, np.arange(1, 367))
        np.testing.assert_allclose(climatology.climatology.sel(dayofyear=60).values, values[59])
        assert int(climatology.sample_count.sel(dayofyear=60)) == 1
    with pytest.raises(ValueError, match="1979--2016"):
        data.prepare_statistics(truth, tmp_path / "invalid", "2018-01-01", "2018-12-31")
