"""Canonical variable order for the 24-hour daily-mean unified experiment."""

PRESSURE_LEVELS = [10, 50, 100, 200, 300, 500, 700, 850, 925, 1000]
PROTOCOL_VARIABLES = [
    (f"{prefix}{level}hpa" if prefix in {"u", "v"} and level == 10 else f"{prefix}{level}")
    for prefix in ("z", "q", "t", "u", "v")
    for level in PRESSURE_LEVELS
] + ["u10", "v10", "t2m", "mslp"]

# The official checkpoints do not expose 10 hPa.  This is the strict real-data
# intersection used for comparisons across all eleven models; local models may
# still train and emit the complete 54-variable S2S protocol above.
OFFICIAL_PRESSURE_LEVELS = PRESSURE_LEVELS[1:]
COMMON_EVALUATION_VARIABLES = [
    f"{prefix}{level}"
    for prefix in ("z", "q", "t", "u", "v")
    for level in OFFICIAL_PRESSURE_LEVELS
] + ["u10", "v10", "t2m", "mslp"]
