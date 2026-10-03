import json

import numpy as np

import src.data_sources.s2s_daily as s2s_daily
from src.experiments.protocol import (
    COMMON_EVALUATION_VARIABLES,
    PRESSURE_LEVELS,
    PROTOCOL_VARIABLES,
)
from src.utils.xarray_utils import CANONICAL_VARIABLES


def test_daily_protocol_has_54_unambiguous_variables():
    assert PRESSURE_LEVELS == [10, 50, 100, 200, 300, 500, 700, 850, 925, 1000]
    assert len(PROTOCOL_VARIABLES) == 54
    assert len(set(PROTOCOL_VARIABLES)) == 54
    assert "u10hpa" in PROTOCOL_VARIABLES
    assert "v10hpa" in PROTOCOL_VARIABLES
    assert "u10" in PROTOCOL_VARIABLES
    assert "v10" in PROTOCOL_VARIABLES


def test_all_protocol_variables_are_canonical():
    assert set(PROTOCOL_VARIABLES).issubset(CANONICAL_VARIABLES)
    assert CANONICAL_VARIABLES["u10hpa"] == {"name": "u_component_of_wind", "level": 10}
    assert CANONICAL_VARIABLES["u10"] == {"name": "10m_u_component_of_wind", "level": None}


def test_official_comparison_uses_real_49_variable_intersection():
    assert len(COMMON_EVALUATION_VARIABLES) == 49
    assert set(COMMON_EVALUATION_VARIABLES).issubset(PROTOCOL_VARIABLES)
    assert not {"z10", "q10", "t10", "u10hpa", "v10hpa"} & set(COMMON_EVALUATION_VARIABLES)


def test_daily_backend_uses_full_horizon_validation_embargo(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        s2s_daily,
        "load_coordinates",
        lambda _root: (
            np.linspace(90.0, -90.0, 121, dtype=np.float32),
            np.arange(240, dtype=np.float32) * 1.5,
        ),
    )
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "variables": list(PROTOCOL_VARIABLES),
                "fit_init_start": "1979-01-01",
                "fit_init_end": "2016-12-31",
                "mean": [0.0] * 54,
                "std": [1.0] * 54,
            }
        )
    )
    backend = s2s_daily.S2SDailyBackend(tmp_path, stats)
    assert backend.validation_init_time[0] == np.datetime64("2017-01-11")
    assert backend.validation_init_time[-1] == np.datetime64("2017-12-31")
    assert len(backend.validation_init_time) == 355
    latest_fit_target = (
        backend.fit_init_time[-1] + np.timedelta64(10, "D")
    )
    assert latest_fit_target < backend.validation_init_time[0]
