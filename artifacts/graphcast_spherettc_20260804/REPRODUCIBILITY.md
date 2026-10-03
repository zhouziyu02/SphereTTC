# 最新实验复现说明

## 1. 推荐：只读确定性复现

从 `SOON` 根目录运行：

```bash
python -m pip install -r requirements-repro.txt
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_latest_results_reproducibility.py
```

这会从冻结逐 initialization 指标重做参数选择、指标聚合、2020 bootstrap、5,880 行 CSV 和 49 张表，不使用 GPU，也不会改写权威结果。`--full-integrity` 额外读取约 14 GB 文件并核对树哈希。

## 2. 复现层级

| 层级 | 本仓库是否自包含 | 入口 |
|---|---|---|
| 原 49 表统计聚合 | 是 | `scripts/verify_main_results_reproducibility.py` |
| 最新结果统计聚合 | 是 | `scripts/verify_latest_results_reproducibility.py` |
| 2017 参数选择 / 2020 schedule / bootstrap | 是 | 最新只读验证器，或各实验 `tools/` |
| 从冻结 GraphCast 缓存重跑 SphereTTC | 是，需要 CUDA GPU | `schedule/tools/run_prospective_holdout.py` 等 |
| 从官方 checkpoint 重跑 2019/2020 GraphCast | 是，需要兼容的 JAX/CUDA | `schedule/tools/run_graphcast_holdout.py` |
| 重新下载 WeatherBench2 truth | 代码可用，但需要网络 | `schedule/tools/prepare_truth_holdout.py`；通常无需执行，truth 已归档 |

## 3. GPU 资产

- checkpoint：`external/official_checkpoints/graphcast/GraphCast_small_1p0deg.npz`
- GraphCast 源码：`external/graphcast_official/`
- 2019 warmup：`schedule/cache/graphcast_2019_current_a100.zarr`
- 2020 forecast：`schedule/cache/graphcast_2020_current_a100.zarr`
- 2020–2021-01-10 truth：`schedule/data/S2S/`
- ACC climatology：`../ttc_publication_corrected_20260726/climatology/DAILY_DOY_1979_2016_COMMON49.npz`

实际运行验证于 Python 3.11、两张 NVIDIA A100-SXM4-80GB；核心包版本记录在 `ENVIRONMENT_20260804.txt`。不同 CUDA/驱动环境应先安装匹配的 `jaxlib`，再以本地 `external/graphcast_official` 安装 GraphCast。

## 4. 安全说明

原实验工具默认把新输出写回各自归档目录。若要做完整 GPU 重跑，建议先复制整个仓库或另建工作副本；权威的日常复现应使用只读验证器。历史 `profiles/`、`logs/` 和旧审计文件中的 `/mnt/...` 绝对路径只是原始 provenance，工具运行路径已改为自动发现迁移后的仓库根目录。
