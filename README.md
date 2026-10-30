# SphereTTC

Minimal review implementation for **Enhancing Medium-range Global Weather Forecasting with Spherical Harmonic Test-Time Calibration**.

SphereTTC calibrates the outputs of a frozen forecaster using truncated spherical harmonics, a closed-form affine fit, and chronological validation gating. It does not load or update the forecasting backbone's weights.

## Contents

```text
src/spherettc/core.py          SphereTTC implementation and online interface
src/spherettc/data.py          ERA5 preparation and training-split statistics
src/spherettc/evaluation.py    Pooled RMSE, ACC, and macro aggregation
configs/params/               Frozen settings for the 11 main backbones
scripts/                      Download, replay, and table-generation commands
tests/                        CPU tests of calibration, timing, data, and metrics
```

No datasets, forecast arrays, checkpoints, generated results, or Git history are included in the review archive. Downloads and outputs are written to ignored directories.

## Reproduction scope

The **calibration and evaluation** pipeline is executable for all 11 backbones when their original forecast caches are available. The original six locally trained backbone checkpoints and the complete eleven-backbone forecast caches are not distributed in this snapshot and do not currently have public download links. Therefore, downloading ERA5 alone does **not** reproduce every number in the main table.

The commands below provide (1) the main-experiment replay procedure, with these prerequisites stated explicitly, and (2) an independently downloadable GraphCast/Pangu example. The public forecast example uses different forecast archives and must not be identified as an exact reproduction of the main table. Retraining a different backbone or substituting a public archive also does not recover the original forecasts.

## 1. Install and test

Use Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

CPU is sufficient for the tests and a small replay. For a full GPU replay, install a PyTorch 2.7.1 CUDA build compatible with your driver, then use `--device cuda`. The package pins the principal dependencies; `torch-harmonics` is pinned to 0.8.0. The original run did not record its harmonic-library version, so bitwise cross-environment agreement is not assumed.

## 2. Download the main-protocol ERA5 verification fields

First inspect the plan without downloading anything:

```bash
python scripts/download_era5.py \
  --source main_daily --start 1979-01-01 --end 2020-01-10 \
  --output data/main_truth.zarr --dry-run
```

Then run the same command without `--dry-run`. Add `--resume` to continue an interrupted download. This includes the fitting period, the 2017 warmup/validation period, both evaluation years, and verification through the final ten-day forecast. The 49-field output is approximately 80 GiB uncompressed; upstream chunks can make network transfer substantially larger. The script loads and writes one day at a time.

The `main_daily` source preserves the recorded daily ERA5 fields and samples the exact 1.5-degree subgrid from their 0.25-degree grid. The optional `conservative_6h` source averages 00/06/12/18 UTC from the public conservative grid and is a separate data protocol. The source URI, variables, dates, and sampling convention are recorded with the generated store. Neither source is silently substituted for the other.

## 3. Build statistics from the fitting split

```bash
python scripts/prepare_statistics.py \
  --truth data/main_truth.zarr \
  --fit-start 1979-01-01 --fit-end 2016-12-31 \
  --output-dir data/main_statistics
```

This creates `stats.json` and `climatology.zarr`. Climatology is a calendar-day mean with 366 month/day slots, including February 29. Normalization uses the population standard deviation over training days and grid cells, with uniform grid weights; evaluation uses cosine-latitude weights. The archived training statistics do not fully record their estimator, so regenerated standard deviations are not guaranteed to match those archived constants bitwise. Exact main-table matching requires the original normalization constants and climatology as well as the original forecasts. Never estimate these inputs from evaluation targets.

## 4. Prepare the original frozen forecast caches

Place one cache per backbone in `data/main_forecasts/`:

```text
convlstm.zarr       transformer.zarr   fno.zarr       vit.zarr
cirt.zarr           climode.zarr       fourcastnetv2.zarr
oneforecast.zarr    fuxi.zarr          pangu.zarr     graphcast.zarr
```

Each cache must contain:

| Item | Contract |
|:--|:--|
| Array | `prediction` (aliases `pred` and `forecast` accepted) |
| Dimensions | `(init_time, lead_time, variable, lat, lon)` |
| Initializations | Daily 00 UTC; a 2017 warmup prefix and all 730 dates in 2018–2019 |
| Lead times | Integer hours `[24, 72, 120, 168, 240]` |
| Variables | 49 common fields, in physical units; extra fields are selected out by name |
| Latitude | 121 points, 90 to −90 degrees |
| Longitude | 240 points, 0 to 358.5 degrees |
| Forecast statistic | Daily mean at each target day; no embedded verification is used |

The 49 fields are geopotential, specific humidity, temperature, and zonal/meridional wind at nine pressure levels (50, 100, 200, 300, 500, 700, 850, 925, 1000 hPa), plus `u10`, `v10`, `t2m`, and `mslp`. Canonical names are defined in `src/spherettc/data.py`. Geopotential is in m² s⁻² and MSLP in Pa; do not substitute geopotential height or hPa values without converting.

The paper's six local architectures were trained from random initialization on 1979–2016, with 2017 used for validation. Their original trained weights are required to regenerate their original caches. The five released systems use the following upstream implementations and assets:

| Backbones | Original weight route |
|:--|:--|
| ConvLSTM, Transformer, FNO, ViT, CirT, ClimODE | Locally trained weights; unavailable in this snapshot |
| FourCastNetV2 | [Official ECMWF wrapper](https://github.com/ecmwf-lab/ai-models-fourcastnetv2), `weights.tar` and paired means/stds |
| OneForecast | [Official implementation](https://github.com/YuanGao-YG/OneForecast), released `20241008-171138` checkpoint and paired normalization |
| FuXi | [Official implementation](https://github.com/tpys/FuXi), original FuXi_EC short/medium/long ONNX release |
| Pangu | [Official implementation](https://github.com/198808xc/Pangu-Weather), 24-hour and 6-hour ONNX models |
| GraphCast | [Official implementation](https://github.com/google-deepmind/graphcast), GraphCast-small 1-degree checkpoint and paired statistics |

This minimal package consumes frozen predictions; it does not provide a replacement backbone training recipe. Upstream model environments, weight terms, and inference inputs remain separate from the calibration environment.

## 5. Replay all main experiments

```bash
python scripts/run_main.py \
  --forecast-dir data/main_forecasts \
  --truth data/main_truth.zarr \
  --statistics-dir data/main_statistics \
  --device cuda --output runs/main
```

Add `--dry-run` to inspect all commands. Use `--models convlstm transformer climode fourcastnetv2 fuxi graphcast` for the six-backbone main-text table, or omit it for all 11. The frozen calibration settings are in `configs/params/`; GraphCast uses `L=64`, strength 0.8, while CirT uses memory 32 and half-life 16. Parameters are not tuned on evaluation targets.

The archived main protocol admits a pair when its target-date timestamp is no later than issuance (`--verification-delay-hours 0`). For daily-mean verification this does not add a delay for completion of the target day. To run a **separate completed-day evaluation**, pass `--verification-delay-hours 24` and a different output directory. Do not mix the two protocols in one table.

Each run calibrates before storing its current pair, filters memory by the chosen availability rule, and requires a full eligible warmup memory. Missing dates, inconsistent grids, non-finite fields, and existing result directories produce explicit errors.

## 6. Inspect the tables

`runs/main/summary/` contains `main_results.csv`, `overall_results.csv`, and `main_results.md`. Each model directory also contains per-variable scores, a chronological memory/gate log, and pooled sufficient statistics. Results are calculated from the evaluated arrays; paper numbers are not hard-coded.

RMSE pools latitude-weighted squared errors over initialization dates before taking the square root. ACC pools anomaly cross-products and both anomaly energies before normalization. For each lead, nRMSE averages `RMSE_v / sigma_v` across variables and ACC averages variable-level ACC values; overall scores then average the five leads. Improvements are:

```text
Delta nRMSE (%) = 100 * (1 - calibrated_nRMSE / raw_nRMSE)
Delta ACC   (%) = 100 * (calibrated_ACC / raw_ACC - 1)
```

Higher improvements are better; negative changes are retained. Relative ACC is left undefined for a nonpositive raw ACC. The cross-model nRMSE improvement is the equally weighted mean of model-level relative improvements, not a ratio of pooled models.

## Downloadable public-data example

This example runs without backbone weights or private caches. It evaluates five variables at 1/3/5/7-day leads on published WeatherBench2 forecasts. Use a bounded window first:

```bash
python scripts/download_forecasts.py \
  --model graphcast --start 2018-01-01 --end 2018-03-31 \
  --output data/example/graphcast.zarr
python scripts/download_era5.py \
  --source conservative_6h --start 2018-01-02 --end 2018-04-07 \
  --variables z500 t850 t2m mslp u10 --output data/example/truth.zarr
```

Download the same five verification variables for 1979–2016 into a separate store, and build statistics using the same source:

```bash
python scripts/download_era5.py \
  --source conservative_6h --start 1979-01-01 --end 2016-12-31 \
  --variables z500 t850 t2m mslp u10 --output data/example/fit_truth.zarr
python scripts/prepare_statistics.py \
  --truth data/example/fit_truth.zarr --output-dir data/example/statistics
```

Then replay:

```bash
python scripts/run_experiment.py \
  --model graphcast --forecast-cache data/example/graphcast.zarr \
  --truth data/example/truth.zarr \
  --climatology data/example/statistics/climatology.zarr \
  --statistics data/example/statistics/stats.json \
  --params configs/params/graphcast.json \
  --variables z500 t850 t2m mslp u10 \
  --start 2018-03-15 --end 2018-03-31 \
  --require-warmup --verification-delay-hours 24 \
  --device cpu --output runs/example_graphcast
python scripts/summarize_results.py \
  --results runs/example_graphcast/results.json --output runs/example_summary
```

Replace `graphcast` with `pangu` in the download, parameter, and replay paths to run the other public archive. These archives end at 240 hours, so daily means requiring samples after 240 hours cannot be constructed; the script rejects ten-day daily means rather than using incomplete days. See the official [WeatherBench2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) for source descriptions and dataset licenses.
