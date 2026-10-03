# GraphCast + SphereTTC main-experiment recommendation

## Decision: `RECOMMEND_INCLUDE_GLOBAL_PRIMARY`

Use the single global `l64_s08` configuration for the GraphCast main
result. Keep the family/lead schedule as an ablation or appendix result;
it is more flexible and therefore less clean as the headline comparison.


## Frozen global parameters

```json
{
  "confidence_z": 0.5,
  "half_life": 32.0,
  "holdout_fraction": 0.25,
  "lmax": 64,
  "max_scale_delta": 0.5,
  "memory_size": 64,
  "min_holdout": 8,
  "min_memory": 24,
  "mmax": 64,
  "ridge": 0.05,
  "robust_quantile_scale": 3.0,
  "strength": 0.8,
  "taper_power": 1.0
}
```

## Evidence

| Evaluation window | Mean cell-relative RMSE gain | RMSE regressions | Macro ACC change | ACC regressions |
|---|---:|---:|---:|---:|
| 2017 search | 3.6631% | 3/245 | +0.00080662 | 49/245 |
| 2017 chronological holdout | 3.2232% | 1/245 | +0.00097878 | 31/245 |
| 2018–2019 post-freeze retrospective | 3.3834% | 2/245 | +0.00154237 | 21/245 |
| 2020 prospective primary | 3.5734% | 0/245 | +0.00256941 | 10/245 |
| 2020 prospective secondary schedule | 3.6758% | 0/245 | +0.00297449 | 1/245 |

For the 2020 primary, the 14-day circular block-bootstrap 95% interval
for mean RMSE gain is [3.4241%, 3.7289%]
and for macro ACC change is [+0.00200672, +0.00308143].

## 2020 primary by lead

| Lead | Mean RMSE gain | RMSE regressions | Macro ACC change | ACC regressions |
|---:|---:|---:|---:|---:|
| 24h | 4.8834% | 0/49 | +0.00139442 | 0/49 |
| 72h | 1.0095% | 0/49 | +0.00104600 | 0/49 |
| 120h | 0.9471% | 0/49 | +0.00160548 | 1/49 |
| 168h | 3.1730% | 0/49 | +0.00392701 | 0/49 |
| 240h | 7.8540% | 0/49 | +0.00487411 | 9/49 |

## Integrity and interpretation

- SphereCast and SphereTTC source code were not edited; all changes
  and outputs are confined to this sibling experiment directory.
- The original source/config/checkpoint hashes match the pre-run snapshot.
- The frozen original main-result verifier still reports
  `MAIN_RESULTS_REPRODUCIBLE`.
- The 2018–2019 check uses the existing frozen GraphCast caches and is
  directly comparable to that protocol. The 2020 result is an additional
  prospective robustness experiment generated from the official local
  GraphCast-small checkpoint on A100 GPUs.
