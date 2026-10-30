"""Checks for pooled metrics and strict field alignment."""

import numpy as np
import pytest
import xarray as xr

from spherettc.evaluation import (
    PooledMetrics,
    calendar_day_numbers,
    macro_scores,
    relative_improvements,
    require_same_coordinates,
    validate_grid,
)


def test_rmse_pools_squared_errors_before_square_root():
    accumulator = PooledMetrics(1, 1, [90, 0, -90])
    target = np.ones((1, 1, 3, 4))
    climatology = np.zeros_like(target)
    for offset in (0, 4):
        prediction = target + offset
        accumulator.update(prediction, prediction, target, climatology)
    rmse = accumulator.scores()["raw"]["rmse"].item()
    assert rmse == pytest.approx(np.sqrt(8))
    assert rmse != pytest.approx(2)


def test_acc_pools_cross_products_instead_of_daily_correlations():
    accumulator = PooledMetrics(1, 1, [90, 0, -90])
    zero = np.zeros((1, 1, 3, 4))
    accumulator.update(zero + 1, zero + 1, zero + 1, zero)
    accumulator.update(zero + 4, zero + 4, zero + 2, zero)
    assert accumulator.scores()["raw"]["acc"].item() == pytest.approx(9 / np.sqrt(17 * 5))


def test_leap_calendar_uses_month_day_without_post_february_shift():
    dates = ["2018-02-28", "2020-02-29", "2018-03-01", "2020-03-01", "2019-12-31"]
    assert calendar_day_numbers(dates).tolist() == [59, 60, 61, 61, 366]


def test_macro_normalizes_per_variable_before_averaging():
    scores = {
        "raw": {"rmse": [[2, 10], [4, 20]], "acc": [[0.5, 0.7], [0.3, 0.5]]},
        "spherettc": {"rmse": [[1, 5], [2, 10]], "acc": [[0.6, 0.8], [0.4, 0.6]]},
    }
    result = macro_scores(scores, [2, 10])
    assert result["raw"]["nrmse_by_lead"] == [1, 2]
    assert result["raw"]["overall_nrmse"] == 1.5
    nrmse, acc = relative_improvements(1.5, 0.75, 0.5, 0.6)
    assert nrmse == 50
    assert acc == pytest.approx(20)
    assert relative_improvements(0, 0, 0, 1) == (None, None)


def test_coordinates_must_match_even_when_shapes_match():
    first = xr.DataArray(np.ones((1, 3, 4)), dims=("variable", "lat", "lon"), coords={"variable": ["t2m"], "lat": [90, 0, -90], "lon": [0, 90, 180, 270]})
    second = first.assign_coords(lat=[-90, 0, 90])
    with pytest.raises(ValueError, match="Coordinates differ for lat"):
        require_same_coordinates(first, second)
    validate_grid(first.lat, first.lon)
    with pytest.raises(ValueError, match="Latitude"):
        validate_grid(second.lat, second.lon)


def test_nonfinite_values_and_zero_anomaly_energy_fail():
    accumulator = PooledMetrics(1, 1, [90, 0, -90])
    zero = np.zeros((1, 1, 3, 4))
    with pytest.raises(ValueError, match="Non-finite"):
        accumulator.update(zero + np.nan, zero, zero, zero)
    accumulator.update(zero, zero, zero, zero)
    with pytest.raises(ValueError, match="zero-energy"):
        accumulator.scores()


def test_statistics_reject_invalid_or_evaluation_split_dates(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_experiment import read_statistics

    path = tmp_path / "stats.json"
    for fit_end in ("NaT", "2018-12-31", None):
        path.write_text(json.dumps({"fit_end": fit_end, "variables": ["t2m"],
                                    "standard_deviation": [1.0]}))
        with pytest.raises(ValueError, match="fit split"):
            read_statistics(str(path), ["t2m"])


def test_cli_replay_delay_and_summary(tmp_path):
    """Exercise the public commands on generated fields, never archived results."""
    import csv
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    lat, lon = np.linspace(90, -90, 9), np.arange(16) * 22.5
    times = np.arange("2018-01-01", "2018-01-14", dtype="datetime64[D]").astype("datetime64[ns]")
    truth_times = np.arange("2018-01-01", "2018-01-17", dtype="datetime64[D]").astype("datetime64[ns]")
    leads = np.array([24, 72])
    field = 280 + np.sin(np.deg2rad(lat))[:, None] + np.cos(np.deg2rad(lon))[None, :]
    target = field[None, None] + np.arange(len(truth_times))[:, None, None, None] / 10
    coordinates = {"variable": ["t2m"], "lat": lat, "lon": lon}
    truth = xr.DataArray(target.astype(np.float32), dims=("time", "variable", "lat", "lon"), coords={"time": truth_times, **coordinates})
    truth.rename("truth").to_dataset().to_zarr(tmp_path / "truth.zarr", mode="w", consolidated=True)
    valid = times[:, None] + leads.astype("timedelta64[h]")[None, :]
    forecast = np.stack([truth.sel(time=dates).values for dates in valid]) + 2
    xr.Dataset({"pred": (("init_time", "lead_time", "variable", "lat", "lon"), forecast)}, coords={"init_time": times, "lead_time": leads, **coordinates}).to_zarr(tmp_path / "forecast.zarr", mode="w", consolidated=True)
    xr.Dataset({"climatology": (("dayofyear", "variable", "lat", "lon"), np.zeros((366, 1, 9, 16), np.float32))}, coords={"dayofyear": np.arange(1, 367), **coordinates}, attrs={"fit_start": "1979-01-01", "fit_end": "2016-12-31"}).to_zarr(tmp_path / "climatology.zarr", mode="w", consolidated=True)
    (tmp_path / "stats.json").write_text(json.dumps({"variables": ["t2m"], "standard_deviation": [10], "fit_end": "2016-12-31"}))
    (tmp_path / "params.json").write_text(json.dumps({"memory_size": 32, "lmax": 4, "mmax": 4, "min_memory": 4, "min_holdout": 2}))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable, str(root / "scripts/run_experiment.py"),
        "--forecast-cache", str(tmp_path / "forecast.zarr"), "--truth", str(tmp_path / "truth.zarr"),
        "--climatology", str(tmp_path / "climatology.zarr"), "--statistics", str(tmp_path / "stats.json"),
        "--params", str(tmp_path / "params.json"), "--model", "fixture",
        "--start", "2018-01-11", "--end", "2018-01-13", "--variables", "t2m",
    ]
    for delay, expected_count in ((0, 8), (24, 7)):
        output = tmp_path / f"delay_{delay}"
        completed = subprocess.run(command + ["--verification-delay-hours", str(delay), "--output", str(output)], env=environment, text=True, capture_output=True, check=True)
        assert json.loads(completed.stdout)["n_initializations"] == 3
        with (output / "replay_log.csv").open() as handle:
            first = next(csv.DictReader(handle))
        assert int(first["minimum_memory_pairs"]) == expected_count
        result = json.loads((output / "results.json").read_text())
        assert result["macro"]["raw"]["overall_nrmse"] == pytest.approx(0.2)
    rejected = subprocess.run(command + ["--require-warmup", "--output", str(tmp_path / "rejected")], env=environment, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "full eligible memory" in rejected.stderr
    assert not (tmp_path / "rejected").exists()
    subprocess.run([sys.executable, str(root / "scripts/summarize_results.py"), "--results", str(tmp_path / "delay_0/results.json"), "--output", str(tmp_path / "tables")], env=environment, check=True, capture_output=True)
    with (tmp_path / "tables/main_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert float(rows[0]["raw_nrmse"]) == pytest.approx(0.2)
    import zarr

    zarr.open_group(tmp_path / "climatology.zarr", mode="a").attrs["fit_end"] = "2018-12-31"
    zarr.consolidate_metadata(tmp_path / "climatology.zarr")
    rejected = subprocess.run(command + ["--output", str(tmp_path / "invalid_climatology")], env=environment, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "pre-2017 training split" in rejected.stderr
