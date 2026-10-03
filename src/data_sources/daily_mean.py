from __future__ import annotations

from collections import defaultdict


DAILY_SAMPLE_OFFSETS_HOURS = (0, 6, 12, 18)


def daily_mean_sample_hours(lead_hours: list[int]) -> dict[int, tuple[int, ...]]:
    if any(int(lead) % 24 for lead in lead_hours):
        raise ValueError(f"daily-mean leads must be divisible by 24 hours: {lead_hours}")
    return {
        int(lead): tuple(int(lead) + offset for offset in DAILY_SAMPLE_OFFSETS_HOURS)
        for lead in lead_hours
    }


def six_hour_step_to_lead_indices(lead_hours: list[int]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = defaultdict(list)
    for lead_index, lead in enumerate(lead_hours):
        for sample_hour in daily_mean_sample_hours(lead_hours)[int(lead)]:
            if sample_hour % 6:
                raise ValueError(sample_hour)
            mapping[sample_hour // 6].append(lead_index)
    return dict(mapping)
