import numpy as np

from src.spherettc import EligibleMemory


def test_no_eligible_history():
    init = np.array(["2020-01-01T00", "2020-01-01T12"], dtype="datetime64[h]")
    valid = init[:, None] + np.array([24, 72], dtype="timedelta64[h]")[None, :]
    mem = EligibleMemory(max_size=32, mode="per_lead")
    assert mem.get_eligible_indices(init[0], init, valid, 0).tolist() == []


def test_valid_time_equal_current_init_is_eligible():
    init = np.array(["2020-01-01T00", "2020-01-02T00"], dtype="datetime64[h]")
    valid = init[:, None] + np.array([24], dtype="timedelta64[h]")[None, :]
    mem = EligibleMemory(max_size=32, mode="per_lead")
    assert mem.get_eligible_indices(init[1], init, valid, 0).tolist() == [0]


def test_valid_time_greater_current_init_is_not_eligible():
    init = np.array(["2020-01-01T00", "2020-01-01T12"], dtype="datetime64[h]")
    valid = init[:, None] + np.array([24], dtype="timedelta64[h]")[None, :]
    mem = EligibleMemory(max_size=32, mode="per_lead")
    assert mem.get_eligible_indices(init[1], init, valid, 0).tolist() == []


def test_different_leads_have_different_eligibility():
    init = np.array(["2020-01-01T00", "2020-01-02T00", "2020-01-04T00"], dtype="datetime64[h]")
    valid = init[:, None] + np.array([24, 72], dtype="timedelta64[h]")[None, :]
    mem = EligibleMemory(max_size=32, mode="per_lead")
    assert mem.get_eligible_indices(init[2], init, valid, 0).tolist() == [0, 1]
    assert mem.get_eligible_indices(init[2], init, valid, 1).tolist() == [0]
