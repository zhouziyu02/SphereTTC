# GraphCast + SphereTTC parameter-schedule exploration

This experiment was originally executed in the isolated sibling
`SOON_graphcast_schedule_20260804` and is now archived inside the portable SOON
repository. Its tools automatically discover the enclosing SOON root; all
experiment caches, truth shards, metrics, logs, profiles, and reports remain
under this directory. Historical absolute paths are provenance only.

## Scientific protocol

- The SphereTTC implementation is not changed. Every run calls the existing
  `../SOON/scripts/run_ttc.py --method sphere_ttc`; only existing command-line
  hyperparameters vary.
- Parameters may differ by forecast lead and by one of six pre-defined
  variable families: geopotential, humidity, temperature, zonal wind,
  meridional wind, and mean-sea-level pressure. SphereTTC fits each variable
  independently, so this is a parameter schedule rather than a new algorithm.
- 2017 is development-only. Initializations 120--273 form the search segment;
  274--364 form a chronological internal holdout.
- A family/lead candidate is eligible on the search segment only if its mean
  RMSE gain is positive, no more than 10% of its variables regress in RMSE,
  and mean ACC does not decrease. Otherwise the no-op setting is available.
- The primary, main-experiment candidate is one global parameter set shared by
  all variables and leads. It must independently reach at least 2% mean RMSE
  gain, non-decreasing macro ACC, and no more than 5% regressed cells on both
  2017 chronological windows. Among eligible settings, maximize the worse of
  the two RMSE gains and break ties by the worse ACC change. This simpler
  primary is separated from the more flexible family/lead schedule, which is
  retained as a pre-specified secondary analysis.
- Both the primary global setting and secondary schedule are frozen before any
  2020 forecast-error metric is computed or inspected.
- The prospective holdout is calendar year 2020. Forecasts are generated from
  the local official GraphCast-small 1-degree checkpoint on an A100 GPU.
  Official WeatherBench2 ERA5 supplies initial conditions and independently
  materialized daily truth through 2021-01-10. Only the final 206
  initializations of 2019 are used as causal SphereTTC warmup; this covers the
  largest searched memory plus forecast-delay retention.
- The prospective success target is at least 2% mean cell-relative RMSE gain,
  non-decreasing macro ACC, and no more than 5% RMSE-regressed cells across
  the common-49 variables and five frozen forecast leads.
- Uncertainty is reported with a fixed 2,000-replicate circular moving-block
  bootstrap over initialization dates (14-day blocks, seed 20260804). This is
  diagnostic uncertainty reporting and does not alter the pass/fail point rule.

The already observed 2018--2019 results are not used to select either setting.
They may be reported later as a secondary comparison, but 2020 is the only
new holdout used for the primary decision.
