# GraphCast + SphereTTC parameter-only GPU result

- Selected on 2017 only: `l24_strength050`
- Frozen test: 2018-2019 (730 daily initializations)
- Mean cell-relative RMSE gain: 1.5482%
- RMSE-regressed cells: 1/245 (0.4082%)
- Macro ACC change: -0.00001117
- ACC-regressed cells: 55/245
- Primary expected effect: FAIL
- Strict non-decreasing macro ACC: FAIL

## By lead

| Lead | Mean RMSE gain | RMSE regressions | Mean ACC change | ACC regressions |
|---:|---:|---:|---:|---:|
| 24h | 2.1942% | 0/49 | +0.00060088 | 0/49 |
| 72h | 0.3529% | 1/49 | +0.00034639 | 0/49 |
| 120h | 0.2008% | 0/49 | +0.00029327 | 3/49 |
| 168h | 1.0537% | 0/49 | +0.00004035 | 21/49 |
| 240h | 3.9392% | 0/49 | -0.00133675 | 31/49 |

The test result is paired against raw GraphCast metrics recomputed in this
isolated directory. Algorithm source and frozen caches remain in `../SOON`.

## Execution and integrity audit

- GPU: two local NVIDIA A100-SXM4-80GB devices; CPU replay was not enabled.
- SphereTTC wall time: 114.13 seconds for 2018 and 116.35 seconds for 2019,
  executed concurrently; peak CUDA memory was 0.376 GB per process.
- The newly computed raw GraphCast arrays are byte-for-byte equal to the
  frozen raw metric arrays for every coordinate, MSE, MAE, bias, and ACC
  covariance component.
- The before/after source hashes and GraphCast-cache metadata fingerprint are
  identical (`integrity/comparison.json`).
- The original repository's deterministic verifier still reports
  `MAIN_RESULTS_REPRODUCIBLE`, including byte-identical reconstruction of its
  frozen main table.

## Interpretation

The conservative parameter set produces a broad RMSE benefit on GraphCast,
but it does not meet the pre-registered 2% mean-gain threshold. The single
reported RMSE regression is numerical equality (`-1.47e-9` relative gain for
q50 at 72h). Macro ACC is effectively flat, although 10-day wind variables
show real ACC reductions; therefore the strict ACC claim is not supported.
