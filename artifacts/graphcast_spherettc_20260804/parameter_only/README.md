# GraphCast + SphereTTC parameter-only GPU experiment

This directory was originally executed as the sibling
`SOON_graphcast_param_only_20260804` and is now archived inside the portable
SOON repository. The tools automatically discover the enclosing SOON root and
reuse its code, GraphCast caches, ERA5 truth, and climatology. Historical logs
and profiles retain their original absolute paths as provenance only.

## Fixed protocol

- Algorithm: the unmodified `SphereTTCCalibrator` selected by
  `../SOON/scripts/run_ttc.py --method sphere_ttc`.
- Backbone: frozen official GraphCast-small 1 degree daily-mean caches.
- Validation/parameter selection: 2017 only. Metrics used for selection start
  after initialization 120 so the causal memory has warmed up.
- Frozen test: 2018 and 2019, opened only after one parameter set is selected.
- Causality: 2018 is warmed with the tail of 2017; 2019 is warmed with the tail
  of 2018. A target enters memory only after its valid time.
- Evaluation: all common-49 variables and 24/72/120/168/240-hour leads;
  cosine-latitude RMSE and fit-split daily-climatology ACC.
- Compute: local NVIDIA GPUs only. No CPU metric-replay flag is permitted.

Only these existing SphereTTC parameters are varied: spherical truncation,
memory/half-life, ridge, correction strength, confidence threshold, maximum
scale delta, and spectral taper. Model code is never copied or edited.

## Pre-registered decision rules

Validation candidates are ranked by mean relative RMSE gain, subject to both:

1. no more than 2% of the 245 variable-by-lead cells may regress in RMSE;
2. macro ACC change must be at least -0.0005.

If no candidate meets both guards, the run is explicitly marked as having no
eligible parameter set and the least-risk fallback is selected for diagnostic
testing. The 2018-2019 test is considered to have the expected primary effect
when mean cell-relative RMSE improves by at least 2% and no more than 1% of
cells regress. Non-decreasing macro ACC is reported as a stricter secondary
criterion rather than silently folded into the RMSE result.

## Reproduction

From this directory, with GPU access:

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/run_experiment.py validation
PYTHONDONTWRITEBYTECODE=1 python tools/run_experiment.py test
```

`tools/run_experiment.py all` runs validation and test in sequence. It uses
both visible GPUs, writes one log and one device profile per task, and never
passes an output path inside `../SOON`.

Current lightweight-package note: these are historical experiment commands and require external per-initialization metrics, forecasts, and truth. Use the repository root `verify_migration.py` for portable read-only verification. Original integrity evidence is under `../../provenance/graphcast_spherettc_20260804/parameter_only/`.
