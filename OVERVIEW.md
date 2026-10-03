# SphereTTC — Project Overview for Co-authors

*Last updated: 2026-10-03 · Maintainer: Ziyu Zhou*

This folder contains the code, frozen results and documentation for **SphereCast**, our project on
**test-time calibration (TTC) for medium-range weather/climate forecasting**. It has two components:

- **SphereTTC**, a plug-and-play test-time calibrator;
- **SphereDyn**, a spherical-dynamics forecasting backbone.

Read this file first. Everything else in the folder is reachable from §11.

**Code at a glance:** the whole SphereTTC module is in **`src/spherettc.py`** and the whole SphereDyn model is in **`src/spheredyn.py`** (one file each, consolidated on 2026-10-03 and verified bitwise-identical to the code that produced the results). Guidance for drawing the framework figures from these two files: **`yaozhong.md`**. Older documents use the internal codename **SOON**; it refers to this project.

---

## 1. TL;DR

| | Claim | Key number (ERA5, test 2018–2019, 730 daily initializations, 49 variables, 1–10-day leads) |
|---|---|---|
| **Result A** | SphereTTC, used as a calibrator, improves existing forecasting models. | Macro nRMSE drops for **11/11 backbones** (6 trained under our protocol + 5 released systems incl. GraphCast, Pangu, FuXi). Mean drop **4.20%**; the average improves at every lead. |
| **Result B** | The spherical principle behind SphereTTC is useful for this task, and SphereDyn is a strong baseline. | SphereDyn beats all 6 same-protocol baselines at every lead on both RMSE and ACC. **SphereDyn + SphereTTC** improves macro nRMSE from 0.3809 to **0.2846 (−25.3%)** and has the **highest macro ACC of all 24 configurations**. It is also **best on both RMSE and ACC at 7 and 10 days**. |

**Status.**

- All numbers are frozen and reproducible from this folder (`python verify_migration.py`).
- Before submission, one causality issue must be fixed and five official-model runs re-run (§9, R1).
- A few ablations and baseline comparisons are still missing (§9).

**Positioning.** To our knowledge, this is the first systematic study of test-time calibration / test-time computing for global medium-range forecasting with modern AI weather models.

- The paper structure follows **PnP-Corrector** (ICML 2026): a new problem formulation, our own SOTA backbone, and a model-agnostic plug-in.
- **ST-TTC** (NeurIPS 2025) introduced test-time computing for spatio-temporal forecasting, but evaluated only on traffic, air-quality and energy data.
- Operational NWP has long used online bias correction (e.g. NCEP decaying-average bias correction, Kalman-filter post-processing, MOS). We must discuss these and compare against them.

---

## 2. Methods

### 2.1 SphereTTC — single-backbone mode (used for Result A)

Code: **`src/spherettc.py`** — Part 1 delayed memory, Part 2 `SphereTTCCalibrator`, Part 3 the online plug-in `SphereTTC`. The experiment runner `scripts/run_ttc.py --method sphere_ttc` imports Parts 1–2 from this file.

At every initialization *t*, for each lead time and each variable, independently:

1. **Delayed memory.** Collect the most recent `memory_size` forecast–truth pairs whose verification time is ≤ *t* (`eligible_from_buffer`). See R1 for the daily-mean subtlety.
2. **Spectral representation.** Transform forecast and truth with a truncated real spherical-harmonic transform (`lmax × mmax`, via `torch_harmonics`).
3. **Conservative affine fit.** Fit `scale · x + bias` per spectral coefficient by weighted ridge regression shrunk toward the identity. Each sample is weighted by **recency** (exponential half-life) and by **robustness** (down-weights anomalous residual energy). The deviation of `scale` from 1 is clipped.
4. **Chronological validation gate.** Refit on the older 75% of memory and test on the newest 25%. The gate equals the optimal blend × a confidence-bounded reliability score × `strength`. A gate of 0 means no change (no-op).
5. **Apply.** Taper the correction by spherical degree ℓ, inverse-transform it and add it to the forecast. The backbone is never modified and there is no gradient training.

Hyper-parameters were selected per backbone on **2017 only** and then frozen:

- `artifacts/ttc_publication_corrected_20260726/params/*.json`
  - local models: lmax 40 (CirT 32)
  - official models: lmax 24
- GraphCast uses a later, also 2017-selected global setting `l64_s08`: lmax 64, strength 0.8.

### 2.2 SphereTTC — multi-provider constrained mode, "v22" (used for Result B)

Code: **`src/spherettc.py`** Part 4 (`fit_constrained_combination`, `apply_constrained_combination`); run by `scripts/spheredyn/run_main_spherettc_v2.py`. Frozen weights: `artifacts/spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/final/FITTED_WEIGHTS_2017.npz`.

- **What it computes.** For each (lead, variable), the calibrated forecast is a constrained linear combination of the target backbone and five released forecasts (FourCastNetV2, OneForecast, FuXi, Pangu, GraphCast), plus a global bias and a shrunk spatial-bias map.
- **Where the weights come from.** Weights are fitted on 2017 with a chronological train/validation split, then frozen. There is no online update during 2018–2019 and no spherical-harmonic component.
- **Actual frozen weights.** SphereDyn receives weight **0.5 in all 245 (lead, variable) cells**. The other 0.5 goes on average to Pangu 0.263, FuXi 0.119, GraphCast 0.064, OneForecast 0.032 and FourCastNetV2 0.021.
- **Link to the proposal.** This is the "constrained combinations with trusted reference systems" component of the proposal. When writing, keep in mind that half of the final forecast comes from external systems (§9, R2).

### 2.3 SphereDyn backbone

Code: **`src/spheredyn.py`**, class `SphereDyn` (`SphereDyn(54, 54, 5)` is exactly the paper model; also built by `src.baselines.build_model("spheredyn_v9_multiscale", …)`). `python -m src.spheredyn` prints the module tree and a parameter breakdown per path.

**Inputs and outputs.**

- **Input:** 3 days of history of 54 normalized variables on a 1.5° grid (121 × 240), plus a seasonal context.
- **Output:** daily-mean fields at 24/72/120/168/240 h.

**Architecture components.**

- A recursive daily flow operator.
- Truncated spherical-spectral band dynamics.
- Spherical depthwise local blocks.
- Semi-Lagrangian transport and harmonic climate tendencies (from v2).
- A direct observation-time pathway (from v7/v8).
- New in v9: a three-scale (native / ½ / ¼) spherical residual pathway.

**Size.**

- **3,966,080 parameters** (verified by strict checkpoint loading).
- Checkpoint: `artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt`, 16.3 MB, SHA-256 `121f9b90…f453`.

**Training recipe recorded in the checkpoint (`args`).**

- Warm-started from a v8 dual-path checkpoint.
- 8 epochs, batch 2, lr 5e-5, weight decay 1e-4, loss = macro RMSE over the common-49 variables.
- 6,000 fit samples, 355 validation samples, seed 44, best epoch 8.

---

## 3. Experimental protocol

| Item | Setting |
|---|---|
| Data | ERA5 on a 1.5° grid (121 × 240), UTC daily means. Official models are initialized at 00 UTC; their daily-mean forecast is the mean of the 00/06/12/18 UTC states of the target day. |
| Variables | common-49: z, q, t, u, v at 50/100/200/300/500/700/850/925/1000 hPa, plus u10, v10, t2m, mslp |
| Leads | 24, 72, 120, 168, 240 h |
| Splits | 1979–2016 for normalization statistics and ACC climatology; **2017 for all hyper-parameter / weight selection**; **2018–2019 for testing** (730 inits); **2020 as an independent prospective holdout** (GraphCast only) |
| Metrics | cos(lat)-weighted RMSE in physical units; ACC against the 1979–2016 daily climatology; **macro nRMSE** = equal-weight mean over 49 variables × 5 leads of RMSE / (1979–2016 std) |
| Backbones | *Trained under our protocol:* ConvLSTM, Transformer, FNO, ViT, CirT, ClimODE-style, **SphereDyn**. *Released systems, frozen and used as-is:* FourCastNetV2, OneForecast, FuXi, Pangu, GraphCast-small 1° |
| Hardware | 11-backbone runs on V100-32GB; GraphCast follow-up and 2020 prospective runs on A100-80GB |

---

## 4. Result A — SphereTTC on 11 backbones (2018–2019)

| Backbone | Raw nRMSE | +SphereTTC | Change | RMSE better / worse cells (of 245) | macro ACC Raw → +SphereTTC |
|---|---:|---:|---:|---:|---:|
| ConvLSTM | 0.4698 | 0.4653 | −0.96% | 237 / 8 | 0.3600 → 0.3722 |
| Transformer | 0.4638 | 0.4589 | −1.05% | 207 / 38 | 0.3881 → 0.3988 |
| FNO | 0.4427 | 0.4403 | −0.55% | 215 / 22 | 0.4424 → 0.4515 |
| ViT | 0.4477 | 0.4465 | −0.27% | 186 / 54 | 0.4321 → 0.4365 |
| CirT | 0.4605 | 0.4601 | −0.10% | 143 / 93 | 0.4156 → 0.4194 |
| ClimODE-style | 0.4798 | 0.4734 | −1.34% | 223 / 22 | 0.3679 → 0.3759 |
| FourCastNetV2 | 0.4500 | 0.3608 | −19.82% | 245 / 0 | 0.6882 → 0.6972 |
| OneForecast | 0.3120 | 0.2994 | −4.06% | 245 / 0 | 0.7740 → 0.7736 |
| FuXi | 0.3253 | 0.2935 | −9.80% | 226 / 13 | 0.7912 → 0.7965 |
| Pangu | 0.2983 | 0.2873 | −3.70% | 243 / 2 | 0.7889 → 0.7883 |
| GraphCast-small 1° | 0.2917 | **0.2784** | −4.57% | 237 / 2 | 0.7988 → 0.8004 |

- **By lead (mean over the 11 backbones):** 24 h −6.28%, 72 h −3.95%, 120 h −2.99%, 168 h −3.34%, 240 h −4.54%. All 11 backbones improve at 24–120 h; 10/11 improve at 168 h and 240 h.
- **ACC:** macro ACC rises on 9/11 backbones; OneForecast and Pangu dip slightly.
- **Best row:** GraphCast + SphereTTC (0.2784) is the lowest macro nRMSE in the whole table.
- **Prospective robustness (GraphCast, 2020, parameters frozen beforehand):**
  - macro nRMSE 0.2946 → 0.2811;
  - **0/245 cells regress**;
  - mean cell RMSE gain 3.57% (14-day block-bootstrap 95% CI [3.42%, 3.73%]);
  - macro ACC +0.0026.
- **Tables:** per-lead tables are in `results/PAPER_TABLE1_NRMSE_BY_LEAD.md` and `results/PAPER_TABLE2_ACC_BY_LEAD.md`.

## 5. Result B — SphereDyn and SphereDyn + SphereTTC (2018–2019)

| Lead | SphereDyn Raw | SphereDyn + SphereTTC | Change | ACC Raw → +TTC | Best same-protocol baseline | GraphCast Raw | +TTC rank (nRMSE / ACC, of 24) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 h | 0.1870 | 0.1274 | −31.9% | 0.932 → 0.968 | 0.3132 (FNO) | 0.1133 | 7 / 7 |
| 72 h | 0.3595 | 0.2271 | −36.8% | 0.724 → 0.905 | 0.4263 (FNO) | 0.1788 | 8 / 8 |
| 120 h | 0.4226 | 0.2922 | −30.9% | 0.583 → 0.838 | 0.4724 (FNO) | 0.2706 | 7 / 7 |
| 168 h | 0.4553 | **0.3532** | −22.4% | 0.478 → **0.746** | 0.4908 (CirT) | 0.3785 | **1 / 1** |
| 240 h | 0.4801 | **0.4231** | −11.9% | 0.369 → **0.583** | 0.5001 (CirT) | 0.5173 | **1 / 1** |
| All | 0.3809 | **0.2846** | **−25.3%** | 0.617 → **0.808** | 0.4427 (FNO) | 0.2917 | 2 / **1** |

- RMSE and ACC both improve in **245/245** cells.
- The frozen main gate (≥ 5% gain over raw SphereDyn) passes: `main_prediction_v2_seed44/MAIN_GOAL_GATE.json`.
- **How to state the SphereDyn claim.** SphereDyn is the best model **among those trained under our protocol**. It is below the four strongest released systems on average. Its lowest raw RMSE at 240 h comes with only the 6th-best ACC, i.e. partly from smoother forecasts. In the paper, split the main table into "same-protocol" and "released systems", as PnP-Corrector does.

## 6. Efficiency

Source: `results/EFFICIENCY.md`, read from the frozen run profiles.

- **SphereTTC cost:**
  - 0 learnable parameters and no gradient training;
  - ≈ **0.07 s** extra per forecast for 49 variables × 5 leads on one A100 (0.36 s end-to-end including truth I/O and scoring);
  - **≈ 0.2% of GraphCast-small inference time**;
  - 2.1 GB peak GPU memory.
- **Hyper-parameter selection:** about 1 min per candidate per backbone (2017, V100).
- **Not yet measured:** MACs/FLOPs, SphereDyn total training GPU-hours, and inference time of the other backbones.

---

## 7. Is all the code here?

Verified on 2026-10-02/03:

- `pytest` passes (38 passed, 1 data-dependent test skipped).
- **Consolidation (2026-10-03):** `src/spherettc.py` and `src/spheredyn.py` reproduce the archived originals exactly. SphereDyn gives bitwise-identical outputs (also with the frozen checkpoint at full resolution). The SphereTTC code is AST-identical. `scripts/run_ttc.py` gives identical metric files before and after for 10 methods. The originals are in `archive/pre_consolidation_20261003/`, and the checks are `tests/test_consolidation_equivalence.py` + `verify_migration.py`.
- The SphereDyn checkpoint loads with `strict=True` and runs a forward pass.
- A synthetic end-to-end SphereTTC run through `scripts/run_ttc.py` removes an injected planetary-scale bias.
- `verify_migration.py` confirms that the 13 key source files on both result paths are identical to the code that produced the frozen numbers, except for path bootstrapping. The one exception is the v22 fitting helper (see the table).

| Component | In this folder? | Notes |
|---|---|---|
| **SphereTTC module (all modes)** | ✅ | **`src/spherettc.py`**; experiment runner `scripts/run_ttc.py`, metrics `src/ttc/metrics.py`; tests `tests/test_sphere_ttc.py`, `tests/test_sphere_ttc_end_to_end.py`, `tests/test_consolidation_equivalence.py` |
| Frozen SphereTTC parameters (11 backbones + `l64_s08`) | ✅ | `artifacts/ttc_publication_corrected_20260726/params/`, `artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json` |
| GraphCast `l64_s08` search / freeze / 2020 prospective tools | ✅ | `artifacts/graphcast_spherettc_20260804/schedule/tools/` |
| SphereTTC multi-provider mode (v22): fit + apply | ✅ / ⚠️ | `src/spherettc.py` Part 4 + frozen weights. The fitting code (moved verbatim from `evaluate_goal_reference_ensemble.py`) differs from the 2026-08-04 runtime record beyond path changes, so it cannot be byte-verified against the run. The apply step and the weights are fine. |
| Experimental SphereTTC variants v2–v7 (multi-timescale / seasonal experts) | ✅ (no results) | `src/ttc/legacy/`. These match the proposal's "multi-timescale and seasonal experts", but **no frozen result uses them**. |
| Comparison TTC methods (static / EMA bias, grid affine, geo, zonal spectral, ST-TTC adapter) | ✅ (no results) | `scripts/run_ttc.py --method …`, `src/ttc/comparators/` |
| **SphereDyn model** | ✅ | **`src/spheredyn.py`**; strict-load verified (`tests/test_spheredyn_checkpoint.py`), bitwise-equal to the original v2→v9 files (`tests/test_consolidation_equivalence.py`) |
| SphereDyn inference + v22 pipeline launchers | ✅ | `scripts/spheredyn/` |
| SphereDyn weights + v22 weights | ✅ | the only two weight files in the folder (≈ 50 MB) |
| **SphereDyn training script** | ❌ | Removed in the 2026-08-03 clean-up. The recipe is recorded in the checkpoint (§2.3), but the v8 warm-start checkpoint and the earlier training chain are not in this folder. The original server archive `SOON_CLEANUP_ARCHIVE_20260803` may still hold them. |
| Raw ERA5 data, forecast caches, per-initialization metrics, GraphCast checkpoint and official model code | ❌ (by design) | Kept on the original server (`/mnt/bn/czl-no-conda/mlx/users/ziyuzhou/SOON`); see `docs/02_REPRODUCE.md` |
| Local baselines and official-model adapters | ✅ | `src/baselines/models.py`, `src/baselines/external_models.py`, `src/data_sources/*_official.py` |
| Result tables / summaries | ✅ | `results/` (regenerated byte-identically by `tools/build_results_summary.py`) |

---

## 8. Repository map

```
SphereTTC/
├── OVERVIEW.md            ← this file (English)
├── yaozhong.md            ← how to draw the SphereTTC / SphereDyn framework figures
├── README.md              ← short landing page (Chinese)
├── verify_migration.py    ← one-command read-only verification (numpy only)
├── requirements.txt       ← SphereTTC runtime + tests; env/ has the other environments
├── src/
│   ├── spherettc.py           ★ SphereTTC — the whole module in one file
│   ├── spheredyn.py           ★ SphereDyn — the whole model in one file
│   ├── ttc/metrics.py         evaluation metrics
│   ├── ttc/comparators/       other TTC methods (geo, zonal spectral, ST-TTC)
│   ├── ttc/legacy/            SphereTTC v2–v7 (no frozen results)
│   ├── baselines/             6 same-protocol baselines + build_model
│   ├── data_sources/          official-model adapters, ERA5 daily loader
│   └── experiments/, utils/
├── scripts/
│   ├── run_ttc.py             ★ entry point for every online TTC method
│   ├── spheredyn/             ★ SphereDyn inference + SphereTTC-v22 + metrics (Result B)
│   ├── tables/                aggregation + 49-variable tables
│   └── preprocessing/         ACC climatology
├── configs/               experiment configs
├── tests/                 unit + smoke tests
├── tools/                 build_results_summary.py
├── results/               ★ paper-ready tables and summaries (generated)
├── docs/                  detailed notes (Chinese), see §11
├── artifacts/             frozen experiment evidence (read-only)
└── archive/               pre-reorganisation snapshot, old READMEs, provenance logs
```

## 9. Known issues and missing experiments (ordered by priority)

Full audit: `docs/04_AUDIT_AND_RISKS.md`.

| # | Issue | Affects | Action / cost |
|---|---|---|---|
| R1 | **Daily-mean memory admission looks ≤ 18 h ahead.** For 00 UTC-initialized official models, a pair is admitted when `valid_time <= init`. Its daily-mean truth, however, still contains 06/12/18 UTC of that day. | Result A, five official backbones (incl. GraphCast `l64_s08`, 2020) | Change admission to `valid_time + 1 day <= init` in `src/spherettc.py::eligible_from_buffer` (single place; `scripts/run_ttc.py` imports it) and `EligibleMemory`, then re-run (≈ 2 min per backbone-year on A100, cached forecasts) |
| R2 | Result B's gain includes the 50/50 combination with released systems | Result B | Run SphereDyn + single-backbone SphereTTC, and the same combination without SphereDyn (cheap) |
| R3 | SphereDyn development used 2018–2019 (the protocol says so explicitly) | Result B | Frozen evaluation on an unused year (2020 or 2023); inference only |
| R7/R12 | No comparison with other TTC / bias-correction methods; "first" must be scoped against NWP online bias correction | Paper | Run `static_bias`, `ema_bias` (≈ decaying-average), `affine_ttc`, `st_ttc` on cached forecasts (cheap) |
| R11 | The proposal's multi-timescale / seasonal experts are not in any final result; v22 has no spherical-harmonic part | Method section | Either drop them from the method text or add results |
| — | Key ablation for "the spherical mechanism helps": SphereTTC with vs without spherical harmonics (`affine_ttc`); SphereDyn component ablation (null control defined in `DESIGN.json`) | Result B narrative | First is cheap; second needs retraining |
| R5 | FourCastNetV2 / FuXi gains concentrate in specific humidity derived from RH | Result A | Check the RH→q conversion; report metrics without q |
| R6 | Two backbones lose a little ACC; the 11 backbones use two selection protocols (publication vs `l64_s08`) | Result A | Re-select all 11 with one 2017 protocol |
| R4, R8–R10 | Test-set reuse history for GraphCast, reproducibility gaps, grid-edge hyper-parameter, evaluation conventions vs WeatherBench2 | Paper | See audit |

All Result A numbers for the five official backbones should be treated as provisional until R1 is fixed.

## 10. How to run

```bash
# Verify all frozen results, tables and code provenance (CPU, numpy only, read-only)
python -m pip install -r env/requirements-verify.txt
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py      # prints SPHERETTC_REPRODUCIBLE ...

# Unit + smoke tests (CPU is enough)
python -m pip install -r requirements.txt
pytest -q
```

Re-running SphereTTC on real forecasts, regenerating GraphCast forecasts, or re-running SphereDyn needs the server-side data and caches. Commands and the asset list are in `docs/02_REPRODUCE.md`. Example (one year of GraphCast + SphereTTC `l64_s08`):

```bash
python scripts/run_ttc.py --config configs/unified_11model_ttc.yaml --model graphcast \
  --cache <graphcast forecast_cache_2018_121x240.zarr> --truth-root <data/S2S> \
  --acc-climatology artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz \
  --method sphere_ttc --params-json artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json \
  --warmup-cache <forecast_cache_2017_121x240.zarr> --trim-warmup-cache --warmup-truth-root <data/S2S> \
  --no-cache-output --metrics-output runs/l64_s08_2018.npz --profile-output runs/l64_s08_2018.json
```

## 11. Further reading and glossary

| Document | Content (language) |
|---|---|
| `yaozhong.md` | Guide for drawing the SphereTTC and SphereDyn framework figures from `src/spherettc.py` and `src/spheredyn.py` (Chinese) |
| `docs/01_SOTA_RESULT.md` | Both results in detail, proposal-claim ↔ evidence table (Chinese) |
| `docs/02_REPRODUCE.md` | Reproduction levels L0–L5, commands, server assets (Chinese) |
| `docs/03_CODE_MAP.md` | Code ↔ result mapping, call chains, version lineage, timeline (Chinese) |
| `docs/04_AUDIT_AND_RISKS.md` | Reviewer-style audit R1–R12 (Chinese) |
| `docs/05_PAPER_STORYLINE.md` | Narrative vs PnP-Corrector, draft contributions (English), experiment plan (Chinese) |
| `docs/PROJECT_PROPOSAL.md` | Project proposal (English) |
| `results/*.md` | Paper-ready tables (English) |
| `artifacts/README.md`, `archive/README.md` | Index of frozen evidence and history (Chinese) |

**Glossary.**

- **SOON**: old internal codename of this project.
- **publication setting**: the per-backbone SphereTTC parameters frozen on 2026-07-26.
- **`l64_s08`**: the later global SphereTTC setting for GraphCast (lmax 64, strength 0.8).
- **v22**: SphereTTC multi-provider constrained mode.
- **cell**: one (variable, lead) pair; 49 × 5 = 245 cells.
- **macro nRMSE**: the primary metric (§3).
- **same-protocol baselines**: models we trained ourselves on the 1.5° daily ERA5 data.
- **released systems**: official pretrained FourCastNetV2, OneForecast, FuXi, Pangu and GraphCast-small, used frozen.
