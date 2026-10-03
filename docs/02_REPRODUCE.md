# 02 · 复现指南

复现分 5 个层级。本仓库自带的内容可以完成 L0–L1；L2 及以上需要原服务器上的数据、缓存和 checkpoint（本包为了轻量没有携带）。

| 层级 | 做什么 | 需要什么 | 本仓库能否独立完成 |
|---|---|---|---|
| L0 | 从冻结 CSV 逐字节重建全部表格与两个 SOTA 的汇总，核对哈希和代码溯源 | Python + numpy | **能** |
| L1 | 单元测试（合成数据） | `requirements.txt`（CPU 即可） | **能** |
| L2 | 用冻结的预报缓存重跑 SphereTTC（SOTA A；以 GraphCast `l64_s08` 的 2017 / 2018–2019 / 2020 为例） | GPU、预报缓存、S2S truth、climatology | 需要外部资产 |
| L3 | 用官方 checkpoint 重新生成 GraphCast 2019/2020 预报 | JAX GPU 环境、GraphCast 源码与 checkpoint、ERA5 | 需要外部资产 |
| L4 | SOTA A 的其余 10 个 backbone | 11 个模型的缓存 | 需要外部资产 |
| L5 | SOTA B：SphereDyn + SphereTTC-v22 | 4 张 GPU、S2S 数据、5 个参考模型缓存 | 需要外部资产 |

## L0 只读验证（推荐先跑）

```bash
python -m pip install -r env/requirements-verify.txt     # 只有 numpy
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py
```

成功时输出以 `SPHERETTC_REPRODUCIBLE` 开头。它不训练、不推理、不写文件，会检查：

1. `MANIFEST.sha256` 中每个文件的 SHA-256；
2. 仓库里没有原始数据或缓存，只有两份允许携带的权重（大小与哈希固定）；
3. 由 5,390 行 baseline CSV + 490 行 SphereDyn CSV 逐字节重建原版 49 张表；由最新 5,880 行 CSV 逐字节重建更新后的 49 张表；新旧之间只有 245 个 GraphCast + SphereTTC 单元变化；
4. 由逐单元 CSV 重算 GraphCast 各实验的汇总，并与冻结 JSON 对照；
5. `results/` 下的全部摘要可以由 `tools/build_results_summary.py` 逐字节再生成，并且仍满足两个 SOTA 的关键结论（SphereTTC 11/11 个 backbone 改善；SphereDyn + SphereTTC 增益 25.28%、245/245 单元改善、macro ACC 全表第一）；
6. 代码溯源：按整理日志反推每个文件，与 2026-08-06 清单和 2026-08-04 运行时源码哈希比对；两个 SOTA 的 13 个关键源文件必须与运行时版本一致（见 `03_CODE_MAP.md` §4）。

## L1 单元测试

```bash
python -m pip install -r requirements.txt
pytest -q
```

2026-10-02 整理后的实测环境：Python 3.13、torch 2.14（CPU）、torch-harmonics 0.8.0、xarray 2024.10.0、zarr 2.18.7、numpy 2.2.6、pandas 2.x，结果为 26 passed、1 skipped（依赖原始 S2S 数据的 loader 测试自动 skip）。同一环境下还确认了 `scripts/run_ttc.py --help` 可用、`src/` 与 `scripts/` 下全部 52 个模块可以导入、两个 4-GPU shell 脚本语法正确。注意 xarray 2025 以后的版本配 zarr 2 会在 `test_run_ttc_daily_acc` 报错，请按 `requirements.txt` 固定版本。

## L2 用冻结缓存重跑 SphereTTC（以 GraphCast `l64_s08` 为例）

需要从原服务器取回（原仓库根：`/mnt/bn/czl-no-conda/mlx/users/ziyuzhou/SOON`；清理归档：`/mnt/bn/czl-no-conda/mlx/users/ziyuzhou/SOON_CLEANUP_ARCHIVE_20260803`）：

| 资产 | 原相对路径 |
|---|---|
| GraphCast 2017–2019 日均预报缓存 | `artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_{2017,2018,2019}_121x240.zarr` |
| 2019 warmup / 2020 预报缓存 | `artifacts/graphcast_spherettc_20260804/schedule/cache/graphcast_{2019,2020}_current_a100.zarr`（迁移前位于 sibling 目录 `SOON_graphcast_schedule_20260804/cache/`） |
| 1979–2020-01-10 S2S truth（1.5°，UTC 日均） | `data/S2S/` |
| 2020–2021-01-10 truth | `artifacts/graphcast_spherettc_20260804/schedule/data/S2S/` |
| ACC climatology | `artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz` |

单年命令（与 `schedule/logs/retrospective_l64_s08_2018.log` 中的原始命令等价，路径改为相对路径；输出写到新建的 `runs/`，不覆盖冻结结果）：

```bash
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/run_ttc.py \
  --config configs/unified_11model_ttc.yaml --model graphcast \
  --cache artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_2018_121x240.zarr \
  --truth-root data/S2S --truth-workers 4 \
  --acc-climatology artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz \
  --no-cache-output --metrics-output runs/l64_s08_2018.npz --profile-output runs/l64_s08_2018.json \
  --method sphere_ttc \
  --params-json artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json \
  --warmup-cache artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_2017_121x240.zarr \
  --trim-warmup-cache --warmup-truth-root data/S2S
```

Raw 对照把 `--method sphere_ttc` 及其后的参数换成 `--method raw`。2019 用 2018 缓存做 warmup。A100 上一年约 2 分钟，峰值显存约 2.1 GB。

整套流程也可以直接用归档里的工具（`search_schedule.py` → `freeze_primary.py` → `run_retrospective_primary.py` → `run_prospective_holdout.py` → `bootstrap_prospective.py` → `finalize_report.py`）。**注意：这些工具会把输出写回 `artifacts/graphcast_spherettc_20260804/` 并覆盖冻结文件**，而且 `run_retrospective_primary.py` 写死了 `CUDA_VISIBLE_DEVICES=1`。请先复制整个仓库到工作副本再运行。

> 修复 R1（`04_AUDIT_AND_RISKS.md`）之后，需要用这一层重跑 2017 选择、2018–2019 和 2020。

## L3 重新生成 GraphCast 预报

```bash
# 先安装与 CUDA 驱动匹配的 jaxlib，再：
python -m pip install -r env/requirements-gpu-graphcast.txt   # 需要 external/graphcast_official/
python artifacts/graphcast_spherettc_20260804/schedule/tools/run_graphcast_holdout.py
python artifacts/graphcast_spherettc_20260804/schedule/tools/prepare_truth_holdout.py   # 需要访问 gs://weatherbench2
```

checkpoint：`external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz`（SHA-256 `e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29`）及其 `stats/`。

## L4 SOTA A 的其余 10 个 backbone

冻结参数：`artifacts/ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json`（每个模型一份，也拆在 `params/*.json`）。用 `scripts/run_ttc.py --method sphere_ttc --params-json params/<model>.json` 逐模型运行，再用 `scripts/tables/summarize_publication_main_table.py` 汇总。本地 6 个 baseline 的缓存与 checkpoint 在原服务器上。

## L5 SOTA B：SphereDyn + SphereTTC-v22

```bash
bash scripts/spheredyn/run_main_prediction_v2_4gpu.sh        # 4 张 GPU：推理 2017/2019 + 拟合 v22 + 门槛判定
bash scripts/spheredyn/compute_spheredyn_main_rmse_acc_4gpu.sh  # RMSE/ACC 指标与 490 行表
```

两个脚本写死了 `/usr/bin/python`，并且需要 `data/S2S`、5 个参考模型的 2017–2019 缓存，以及筛选阶段留下的 2018 SphereDyn 缓存。checkpoint 与拟合权重在本包内：

- `artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt`（16,302,127 字节）
- `artifacts/spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/final/FITTED_WEIGHTS_2017.npz`（35,588,308 字节）

## 原服务器全量归档的验证器

`archive/full_archive_verifiers/` 里的三个脚本（`verify_main_results_reproducibility.py`、`verify_latest_results_reproducibility.py`、`build_latest_reproducibility_manifest.py`）需要约 14 GB 的逐 initialization 指标和缓存，在本包里无法运行。它们保持原样，要用时请复制回原服务器归档的 `scripts/` 目录。

## 已验证的运行环境（2026-08-04）

见 `artifacts/graphcast_spherettc_20260804/ENVIRONMENT_20260804.txt`：Python 3.11，2 × A100-SXM4-80GB，torch 2.7.1，jax/jaxlib 0.4.28，xarray 2024.10.0，zarr 2.18.7，numpy 1.26.4。publication 11 模型结果是在 V100-SXM2-32GB 上跑的。
