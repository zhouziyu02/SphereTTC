# 02 · SphereTTC 复现指南

当前代码库只包含单 backbone SphereTTC 的 11 模型实验。复现分为冻结记录校验、合成测试、真实缓存重放和预报再生成；通过前两项不代表已经完成真实数据复现，也不解决日均 memory 的 R1 问题。

| 层级 | 内容 | 所需资产 | 本仓库独立完成 |
|---|---|---|---|
| L0 | 重建 11 backbone / 22 配置 / 5,390 行的表格与摘要，校验清单与保留源码等价性 | Python + numpy | 能 |
| L1 | 单元测试、合成端到端运行、SphereTTC 等价性检查 | `requirements.txt`，CPU 即可 | 能；外部数据测试可跳过 |
| L2 | 从冻结缓存重跑 SphereTTC 与 Raw 对照 | GPU、forecast cache、S2S truth、ACC climatology | 需外部资产 |
| L3 | 用官方 checkpoint 再生成 GraphCast 预报 | 匹配的 JAX/CUDA 环境、官方源码和 checkpoint、ERA5 | 需外部资产 |
| L4 | 对其余 10 个 backbone 重放统一实验 | 对应缓存 / checkpoint、truth、冻结参数 | 需外部资产 |

## L0 · 只读验证

```bash
python -m pip install -r env/requirements-verify.txt
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py
```

验证器用于检查保留文件的完整性、从冻结 CSV 重建表格、重算 SphereTTC 汇总、核对 GraphCast 记录及保留的源码等价性证据。publication 表与最新表均为 5,390 行；最新表替换了 GraphCast 的 245 个 calibrated 单元。`scripts/tables/build_latest_results.py` 从 publication aggregate CSV 和冻结 GraphCast extract 重建最新表格，不依赖逐 initialization NPZ；`results/` 由 `tools/build_results_summary.py` 生成。历史 integrity / AUDIT 记录位于 `artifacts/provenance/graphcast_spherettc_20260804/`，不作为当前可执行验证工具。

SHA-256、源码等价性和摘要一致性不能证明 R1 已修复、测试集从未参与开发、原始数据正确或真实预测已重新运行。完整实验仍需 L2 及以上。

## L1 · 安装与单元测试

推荐 Python 3.11，与冻结运行记录的主要环境一致：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
python scripts/run_ttc.py --help
```

`requirements.txt` 采用 xarray 2024.10.0 / zarr 2.18.7 的组合，避免历史测试中较新 xarray 与 zarr 2 的兼容性问题。`torch_harmonics` 是核心依赖；原运行版本未记录，历史单元测试在 0.8.0 下通过。实际通过 / 跳过数量以当前运行输出为准，不沿用整理前测试数量。

## L2 · 从缓存重跑 SphereTTC

原服务器根目录为 `/mnt/bn/czl-no-conda/mlx/users/ziyuzhou/SOON`。以下资产没有随轻量仓库提供，需从有权限的原服务器获取：

| 资产 | 原相对路径 |
|---|---|
| GraphCast 2017–2019 日均预报 | `artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_{2017,2018,2019}_121x240.zarr` |
| 2019 warmup / 2020 预报 | `artifacts/graphcast_spherettc_20260804/schedule/cache/graphcast_{2019,2020}_current_a100.zarr`；迁移前为 sibling 目录 `SOON_graphcast_schedule_20260804/cache/` |
| 1979–2020-01-10 S2S truth | `data/S2S/` |
| 2020–2021-01-10 truth | `artifacts/graphcast_spherettc_20260804/schedule/data/S2S/` |
| ACC climatology | `artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz` |
| normalization statistics（已保留） | `artifacts/shared/s2s_daily_54var_stats.json` |

下面重放 2018 GraphCast `l64_s08`，输出到新的 `runs/`，不覆盖冻结指标。truth 和 climatology 须预先放到相应位置。此命令保留历史协议；不能通过重放本身修复 R1。

```bash
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/run_ttc.py \
  --config configs/unified_11model_ttc.yaml --model graphcast \
  --cache artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_2018_121x240.zarr \
  --truth-root data/S2S --truth-workers 4 \
  --acc-climatology artifacts/ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz \
  --method sphere_ttc \
  --params-json artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json \
  --warmup-cache artifacts/ttc_publication_corrected_20260726/official_daily_mean/cache/graphcast/forecast_cache_2017_121x240.zarr \
  --trim-warmup-cache --warmup-truth-root data/S2S \
  --no-cache-output --metrics-output runs/l64_s08_2018.npz \
  --profile-output runs/l64_s08_2018.json
```

Raw 对照改为 `--method raw`，去掉 `--params-json` 与 warmup 参数，并使用不同输出文件名。2019 使用 2018 缓存 warmup。冻结 profile 中，一年 SphereTTC 在 A100 上约 2 分钟，峰值显存约 2.1 GB；真实耗时取决于 I/O 和硬件。

GraphCast 完整工具链位于 `artifacts/graphcast_spherettc_20260804/schedule/tools/`：`search_schedule.py` → `freeze_primary.py` → `run_retrospective_primary.py` → `run_prospective_holdout.py` → `bootstrap_prospective.py` → `finalize_report.py`。

**工具会向 `artifacts/graphcast_spherettc_20260804/` 写回并可能覆盖冻结证据。** 先制作实验工作副本并检查工具中的路径 / GPU 配置；`run_retrospective_primary.py` 历史上固定 `CUDA_VISIBLE_DEVICES=1`。新实验应保留独立输出目录与新清单，不把新结果冒充原冻结记录。

修复 R1 后，重新执行 2017 选参、2018–2019 测试和 GraphCast 2020 评估，并记录 truth availability 的定义及业务观测延迟。五个官方模型均受影响；六个本地日均输入模型也须明确预报发布时刻。

## L3 · 重新生成 GraphCast 预报

```bash
# 先安装与 CUDA 驱动匹配的 jaxlib，再安装官方模型环境。
python -m pip install -r env/requirements-gpu-graphcast.txt
python artifacts/graphcast_spherettc_20260804/schedule/tools/run_graphcast_holdout.py
python artifacts/graphcast_spherettc_20260804/schedule/tools/prepare_truth_holdout.py
```

需要 `external/graphcast_official/`、可访问的 ERA5/WeatherBench2 数据，以及 `external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz` 和相应 `stats/`。冻结 checkpoint 的 SHA-256 为 `e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29`。准备 truth 的脚本需要访问 `gs://weatherbench2`。这些工具同样应在实验工作副本中运行。

## L4 · 其余 10 个 backbone

使用 `scripts/run_ttc.py --method sphere_ttc --params-json artifacts/ttc_publication_corrected_20260726/params/<model>.json`，替换 `--model`、预测缓存与 warmup。模型列表见 `configs/unified_11model_ttc.yaml`；变量、年份、时效与网格见 `configs/unified_11model_protocol.yaml`。用 `scripts/tables/summarize_publication_main_table.py` 汇总逐 initialization 指标。

六个本地 backbone 的缓存与 checkpoint、其他官方模型缓存仍在原服务器。这里保留了模型实现 / adapter，但没有承诺所有 backbone 都能仅用本仓库从头训练和生成预测。补对照实验时，`static_bias`、`ema_bias`、`affine_ttc` 等须与 SphereTTC 使用相同的可用真值、warmup、选参年份和指标协议。

## 冻结运行环境

`artifacts/graphcast_spherettc_20260804/ENVIRONMENT_20260804.txt` 记录：Python 3.11、2 × A100-SXM4-80GB、torch 2.7.1、jax/jaxlib 0.4.28、xarray 2024.10.0、zarr 2.18.7、numpy 1.26.4。publication 11 模型实验使用 V100-SXM2-32GB。GraphCast 2018–2019 缓存和 2020 缓存来自不同 GPU 代际，复现报告应保留这一差异。
