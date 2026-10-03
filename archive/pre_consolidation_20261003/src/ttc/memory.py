from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EligibleMemory:
    max_size: int | None = None
    mode: str = "per_lead"

    def get_eligible_indices(self, current_init_time, init_times, valid_times, lead_index: int | None = None):
        current = np.datetime64(current_init_time)
        init_times = np.asarray(init_times, dtype="datetime64[ns]")
        valid_times = np.asarray(valid_times, dtype="datetime64[ns]")
        if valid_times.ndim != 2:
            raise ValueError("valid_times must have shape [N, L].")
        if self.mode == "per_lead":
            if lead_index is None:
                raise ValueError("lead_index is required for per_lead memory.")
            eligible = np.where(valid_times[:, int(lead_index)] <= current)[0]
            eligible = eligible[init_times[eligible] < current]
            if self.max_size is not None:
                eligible = eligible[-int(self.max_size) :]
            return eligible
        if self.mode == "cross_lead":
            sample_idx, lead_idx = np.where(valid_times <= current)
            keep = init_times[sample_idx] < current
            pairs = list(zip(sample_idx[keep].tolist(), lead_idx[keep].tolist()))
            pairs.sort(key=lambda x: valid_times[x[0], x[1]])
            if self.max_size is not None:
                pairs = pairs[-int(self.max_size) :]
            return pairs
        raise ValueError(f"Unknown memory mode: {self.mode}")


def assert_no_future_targets(current_init_time, selected_valid_times):
    current = np.datetime64(current_init_time)
    selected = np.asarray(selected_valid_times, dtype="datetime64[ns]")
    if selected.size and np.any(selected > current):
        raise AssertionError("TTC memory leakage: selected target valid_time is in the future.")
