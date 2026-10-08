# SphereTTC — Project Overview

*Updated: 2026-10-08*

This repository focuses on **single-backbone test-time calibration for global medium-range weather forecasting**. SphereTTC corrects a frozen forecast using historical forecast–truth pairs. The core implementation is `src/spherettc.py`; `scripts/run_ttc.py` is the experiment runner. Older experiment records use the internal project codename SOON.

The retained main experiment covers 11 backbones, 22 raw/calibrated configurations and 5,390 variable–lead metric rows. Existing frozen numbers remain unchanged by this repository cleanup. They are evidence under the original protocol, **not yet final causal-deployment results**: the daily-mean memory timing issue below must be resolved and the affected runs repeated.

## 1. Method

For each initialization, lead and variable:

1. Gather recent forecast–truth pairs admitted by delayed memory. The implementation currently checks `valid_time <= init`; daily-mean targets require the additional availability delay discussed in R1.
2. Transform forecasts and truth with a truncated real spherical-harmonic transform (`torch_harmonics`).
3. Fit spectral affine corrections with weighted ridge regression shrunk toward identity. Recency weighting tracks changes; robust weighting reduces anomalous residual influence; scale deviation is clipped.
4. Refit on the older memory subset and validate on the more recent subset (default 75% / 25%). Use the optimal blend, a confidence-bounded reliability score and `strength` to determine the correction gate. A zero gate gives a no-op.
5. Taper the correction by spherical degree, inverse-transform it and add it to the forecast.

The backbone remains frozen and no gradient training is required. Calibration coefficients are fitted online. Multiple timescale and seasonal variants in `src/ttc/legacy/` are experimental; none produced the frozen main results.

Publication parameters were selected separately per backbone on 2017 and are stored in `artifacts/ttc_publication_corrected_20260726/params/`. Local models generally use lmax 40 (CirT 32); released models originally use lmax 24. The current GraphCast row uses the later 2017-selected global setting `l64_s08` (lmax 64, strength 0.8). Selection protocols differ, and the decision to expand GraphCast's search followed inspection of 2018–2019 results.

## 2. Protocol

| Item | Setting |
|---|---|
| Data | ERA5 UTC daily means, 1.5° grid, 121 × 240 |
| Released forecasts | Initialize at 00 UTC; target daily mean averages the target day's 00/06/12/18 UTC states |
| Variables | z, q, t, u, v at 50/100/200/300/500/700/850/925/1000 hPa, plus u10, v10, t2m, mslp: common-49 |
| Leads | 24, 72, 120, 168, 240 h |
| Splits | 1979–2016 normalization / ACC climatology; 2017 parameter selection; 2018–2019 evaluation (730 daily initializations); GraphCast 2020 prospective holdout |
| Metrics | cos(lat)-weighted physical RMSE; ACC against daily climatology; macro nRMSE = equal-weight mean of RMSE / training-period standard deviation over 49 variables × 5 leads |
| Local trained backbones | ConvLSTM, Transformer, FNO, ViT, CirT, ClimODE-style |
| Frozen released systems | FourCastNetV2, OneForecast, FuXi, Pangu, GraphCast-small 1° |
| Normalization statistics | `artifacts/shared/s2s_daily_54var_stats.json` |
| Hardware of recorded runs | V100-32GB for publication runs; A100-80GB for GraphCast follow-up / 2020 |

These daily-mean, 1.5° results cannot be directly compared to WeatherBench2 instantaneous 6-hour headline numbers or to the full 0.25° GraphCast model.

## 3. Frozen results and their limits

On 2018–2019, all 11 backbones improve in macro nRMSE under the recorded protocol: mean relative reduction 4.20%, median 1.34%, range 0.10%–19.82%. Mean changes across backbones are −6.28%, −3.95%, −2.99%, −3.34% and −4.54% at the five leads. At 168 and 240 h only 10/11 individual backbones improve.

Macro ACC improves on 9/11 backbones; OneForecast and Pangu regress slightly. Some variable–lead cells regress even when the macro average improves. GraphCast + SphereTTC has the lowest macro nRMSE among the retained 22 configurations: 0.2917 raw → 0.2784 calibrated (−4.57%). Detailed numbers and cell counts are in `docs/01_SOTA_RESULT.md` and `results/SPHERETTC_11_BACKBONES.md`.

The frozen GraphCast 2020 evaluation reports macro nRMSE 0.2946 → 0.2811, no regressing RMSE cells out of 245, mean cell RMSE gain 3.57% (14-day block-bootstrap 95% CI [3.42%, 3.73%]) and macro ACC +0.0026. It is a separate-year evaluation with parameters frozen beforehand; it shares the daily-mean admission problem and must also be rerun after the fix. Do not rank its 2020 numbers against the other models' 2018–2019 results.

Recorded GraphCast profiles give about 0.07 s calibration overhead per initialization on an A100 (49 variables × 5 leads), 0.36 s including truth I/O and scoring, and 2.1 GB peak GPU memory. The incremental cost is about 0.2% of the recorded GraphCast-small inference time. These are profile differences on this hardware, not universal timings. MACs/FLOPs and inference timings of the other backbones are not available. See `results/EFFICIENCY.md`.

## 4. What is reproducible here?

The repository contains the calibrator, runner, metrics, local backbone implementations, official-model adapters, frozen parameter JSONs, aggregate metric CSVs and GraphCast selection / prospective records; historical integrity / audit records are under `artifacts/provenance/`. Unit tests exercise synthetic cases and retained pre-consolidation SphereTTC equivalence. `verify_migration.py` checks repository integrity and reconstruction of frozen summaries; it does not rerun forecasts or validate causality.

Raw ERA5/S2S data, forecast caches, per-initialization metrics, official model source trees and checkpoints are external assets. Consequently, frozen-table verification and synthetic tests are self-contained, while real-data recalibration or forecast regeneration needs the original server assets listed in `docs/02_REPRODUCE.md`. The historical runtime version of `torch_harmonics` was not recorded.

## 5. Repository map

```text
SphereTTC/
├── README.md                quick start, installation and main runner
├── OVERVIEW.md              this overview
├── visualization.md         guide for the SphereTTC framework figure
├── verify_migration.py      read-only integrity and frozen-result checks
├── requirements.txt, env/   runtime, tests and optional model environments
├── src/
│   ├── spherettc.py         delayed memory, spectral calibrator, online plugin
│   ├── ttc/metrics.py       latitude-weighted evaluation
│   ├── ttc/comparators/     comparison calibration methods
│   ├── ttc/legacy/          experimental variants without frozen main results
│   ├── baselines/          six local backbone implementations / adapters
│   ├── data_sources/       ERA5 loader and official-model adapters
│   └── experiments/, utils/
├── scripts/run_ttc.py       shared calibration / evaluation entry point
├── scripts/tables/          aggregate metrics / rebuild retained CSV tables
├── scripts/preprocessing/   ACC climatology preparation
├── configs/                 11-model protocol and execution settings
├── tests/                   unit, synthetic integration and equivalence tests
├── tools/                   deterministic result-summary generation
├── results/                 generated paper-table drafts and summaries
├── docs/                    detailed reproduction, audit and experiment plan
├── artifacts/               retained SphereTTC evidence and shared statistics
└── archive/                 minimal SphereTTC equivalence / provenance evidence
```

## 6. Priorities before making final claims

1. **R1: truth availability.** A target day's mean includes observations later than its 00 UTC timestamp. Admission by `valid_time <= init` can therefore expose up to 18 h of future sampled observations for the five released models. Use an explicit daily-mean availability delay, update memory tests, then rerun 2017 selection, 2018–2019 and GraphCast 2020. Local daily-input baselines also need a clearly stated issue time.
2. Compare static bias, EMA bias, grid affine and existing TTC methods under the same memory / availability protocol. Existing code is not evidence of comparative performance.
3. Measure spherical representation, gate, recency, robustness and taper contributions with controlled ablations; harmonize parameter selection across all 11 backbones.
4. Audit RH→q conversion in FourCastNetV2 / FuXi and report metrics without q; report ACC exceptions, per-cell regressions and GraphCast's search history.
5. Verify related literature before any novelty claim. The current evidence does not establish that SphereTTC is the first approach or state of the art relative to other calibration methods.

Full risk details retain their historical identifiers in `docs/04_AUDIT_AND_RISKS.md`.

## 7. Reading and running

Start with `README.md` for installation, `pytest -q`, `python verify_migration.py` and the main runner. Then read:

| Document | Purpose |
|---|---|
| `docs/01_SOTA_RESULT.md` | Frozen 11-backbone results and claim-to-evidence mapping; historical filename retained |
| `docs/02_REPRODUCE.md` | Reproduction levels, commands, external data / checkpoint paths |
| `docs/03_CODE_MAP.md` | Call chains, experimental variants and provenance limits |
| `docs/04_AUDIT_AND_RISKS.md` | Remaining methodological and reproducibility risks |
| `docs/05_PAPER_STORYLINE.md` | SphereTTC contributions and concrete future experiments |
| `docs/PROJECT_PROPOSAL.md` | Research scope aligned with the retained method |
| `visualization.md` | Framework-diagram guide |

A **cell** is one variable–lead pair (49 × 5 = 245). **Publication setting** means the per-backbone parameters frozen on 2026-07-26; **l64_s08** means the later global GraphCast setting. **Released systems** are pretrained models used frozen; the **local backbones** were trained under the local daily-mean protocol. These two groups do not share a training budget.
