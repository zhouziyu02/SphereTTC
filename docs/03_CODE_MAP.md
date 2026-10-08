# 03 · SphereTTC 代码与证据地图

## 1. 主路径

| 路径 | 角色 |
|---|---|
| `src/spherettc.py` | 完整单 backbone SphereTTC：延迟核验 memory、`SphereTTCCalibrator`、在线插件 `SphereTTC` |
| `scripts/run_ttc.py` | 统一缓存评估入口；选择 Raw / SphereTTC / 对照 / 实验变体，处理 warmup、memory 与 RMSE/ACC |
| `src/ttc/metrics.py` | 纬度加权指标 |
| `src/utils/` | 配置、设备、时间、climatology 下标和 zarr/xarray 工具 |
| `src/data_sources/s2s_daily.py` | 1.5° UTC 日均 truth loader |
| `src/data_sources/*_official.py`、`daily_mean.py` | 五个官方模型的 adapter 与日均预报生成 |
| `src/experiments/protocol.py` | common-49 变量及气压层定义 |
| `configs/unified_11model_ttc.yaml` | TTC 执行配置 |
| `configs/unified_11model_protocol.yaml` | 11 模型统一评估协议 |
| `configs/official_daily_common49_2018_2019.yaml` | 官方日均缓存生成配置 |
| `src/baselines/models.py` | ConvLSTM / FNO / Transformer / ViT 及 `build_model` |
| `src/baselines/external_models.py` | CirT / ClimODE 封装 |
| `src/ttc/comparators/` | geo、zonal spectral、ST-TTC adapter；无冻结主对照结果 |
| `src/ttc/legacy/` | v2–v7 实验校准器；无冻结主结果 |
| `scripts/tables/` | `summarize_publication_main_table.py` 汇总逐 initialization 指标；`build_latest_results.py` 从保留的 aggregate CSV 重建最新 5,390 行与 49 张变量表 |
| `scripts/preprocessing/build_daily_dayofyear_climatology.py` | 1979–2016 逐日 ACC climatology |
| `tools/build_results_summary.py` | 从保留的冻结证据生成 `results/` 摘要 |
| `verify_migration.py` | 只读完整性、表格和保留源码等价性校验 |
| `tests/` | 单元、合成端到端、延迟记忆和 SphereTTC 等价性检查 |
| `artifacts/shared/s2s_daily_54var_stats.json` | 归一化统计，供 common-49 汇总使用 |
| `archive/pre_consolidation_20261003/src/ttc/` | 合并前的 `sphere.py`、`memory.py`，保留作 SphereTTC 等价性参照 |

## 2. SphereTTC 调用链

```text
scripts/run_ttc.py --method sphere_ttc --params-json <frozen params>
  ├── src/spherettc.py
  │     ├── eligible_from_buffer          per-lead delayed memory（R1 未修复）
  │     └── SphereTTCCalibrator           SHT → affine fit → gate → taper → inverse SHT
  ├── src/data_sources/s2s_daily.py       truth
  └── src/ttc/metrics.py                  cos(lat) RMSE / ACC
```

应用集成可使用同一模块的 `SphereTTC` 在线插件。图解见 `../visualization.md`。`--method raw` 走同一评价口径而不校准；其他方法共享的 memory 规则也应在 R1 修复时同步验证。

GraphCast `l64_s08` 的专用工具位于 `artifacts/graphcast_spherettc_20260804/schedule/tools/`：

```text
search_schedule.py               2017：56 候选、search/holdout 两段
  → analyze_validation.py
  → freeze_primary.py            冻结 l64_s08
  → run_retrospective_primary.py  2018–2019，前一年缓存 warmup
  → run_graphcast_holdout.py      生成 2019 warmup / 2020 预报
  → prepare_truth_holdout.py      2020-01-01..2021-01-10 truth
  → run_prospective_holdout.py    2020 校准
  → bootstrap_prospective.py     14 天 block bootstrap
  → finalize_report.py
```

校准阶段通过子进程调用 `scripts/run_ttc.py`。这些专用工具的输出路径可能指向冻结 artifact 目录，只应在实验工作副本中运行。`parameter_only/tools/` 保留了较早负结果的工具，不能用它替代主表的 `l64_s08` 设置。

## 3. 核心实现与实验变体

| 名称 | 内容 | 证据地位 |
|---|---|---|
| `sphere_ttc` | 截断球谐、鲁棒加权仿射、chronological gate、degree taper | 11 backbone 主结果、GraphCast `l64_s08` 和 2020 结果使用的唯一主方法 |
| v2 | 分频带、分分量风险控制 | 实验代码，无冻结主结果 |
| v3 | 频带共享与 validation guard | 同上 |
| v4 | 短 / 长记忆专家混合 | 同上 |
| v5 | 三时间尺度与受保护 no-op | 同上 |
| v6 | 四专家、保守 bias anchor、block-robust | 同上 |
| v7 | 季节残差 memory | 同上 |

不带版本号的 `sphere_ttc` 才是冻结主结果的方法。不能因为变体代码存在，就声称主结果包含多时间尺度 / 季节专家。实现比较方法也不等于已有方法比较证据。

## 4. 校验与溯源边界

2026-10-03 将原 `src/ttc/sphere.py` 与 `src/ttc/memory.py` 的核心合并到 `src/spherettc.py`。保留的两个原文件用于检查计算逻辑等价性；当前校验范围以 `verify_migration.py` 与 `tests/test_consolidation_equivalence.py` 为准。2026-10-08 的清理进一步收敛了代码库范围，不能再把当前仓库描述为历史全量迁移包的完整副本。

冻结参数、metric CSV、GraphCast selection / profile 记录是现有数字的证据。历史 integrity / AUDIT 记录已移至 `artifacts/provenance/graphcast_spherettc_20260804/`；旧 `integrity.py` 不再作为当前入口。清单检查只证明当前保留文件与清单一致；AST / 行为等价性只证明被检查部分未改变，均不能证明真值可用时刻正确或替代原服务器上的真实数据重跑。

原始运行记录中的服务器路径和旧项目名称属于历史证据，不是当前目录布局。对照现代入口时，以本页和 `02_REPRODUCE.md` 为准。任何新实验应另存配置、输出与运行清单。

## 5. SphereTTC 证据时间线

| 日期（2026） | 事件 | 保留证据 |
|---|---|---|
| 07-23 | 本地 baseline 2017 验证缓存与参数精调 | 完整缓存仍在原服务器 |
| 07-26 | publication 参数冻结，2018–2019 评估 | `artifacts/ttc_publication_corrected_20260726/` |
| 08-04 | GraphCast parameter-only 负结果、扩大搜索、`l64_s08`、2020 prospective | `artifacts/graphcast_spherettc_20260804/` |
| 10-03 | SphereTTC 核心合并为单文件 | `src/spherettc.py`、保留的原 `sphere.py` / `memory.py` |
| 10-08 | 仓库聚焦 SphereTTC 与 11 backbone 校准 | 当前入口、配置、结果摘要和校验器 |
