# Prospective 2020 GraphCast + SphereTTC

## Primary: one global parameter set

- Frozen candidate: `l64_s08`
- Result: PASS
- Mean cell-relative RMSE gain: 3.5734%
- RMSE-regressed cells: 0/245
- Macro ACC change: +0.00256941
- ACC-regressed cells: 10/245

| Lead | Mean RMSE gain | RMSE regressions | Mean ACC change | ACC regressions |
|---:|---:|---:|---:|---:|
| 24h | 4.8834% | 0/49 | +0.00139442 | 0/49 |
| 72h | 1.0095% | 0/49 | +0.00104600 | 0/49 |
| 120h | 0.9471% | 0/49 | +0.00160548 | 1/49 |
| 168h | 3.1730% | 0/49 | +0.00392701 | 0/49 |
| 240h | 7.8540% | 0/49 | +0.00487411 | 9/49 |

## Secondary: family/lead parameter schedule

- Result: PASS
- Mean cell-relative RMSE gain: 3.6758%
- RMSE-regressed cells: 0/245
- Macro ACC change: +0.00297449
- ACC-regressed cells: 1/245

| Lead | Mean RMSE gain | RMSE regressions | Mean ACC change | ACC regressions |
|---:|---:|---:|---:|---:|
| 24h | 5.4671% | 0/49 | +0.00156447 | 0/49 |
| 72h | 1.2503% | 0/49 | +0.00121939 | 0/49 |
| 120h | 1.1427% | 0/49 | +0.00180360 | 0/49 |
| 168h | 3.6036% | 0/49 | +0.00405949 | 0/49 |
| 240h | 6.9151% | 0/49 | +0.00622551 | 1/49 |

Both settings were frozen from 2017 before any 2020 forecast-error
metric was computed or inspected. Both 2019 warmup and 2020 forecasts
were generated from the official local checkpoint on the same pair
of A100 GPUs.
