# 03 · 代码地图：哪段代码产出了哪个结果

## 1. 目录与角色

| 路径 | 角色 | 属于哪个 SOTA（A = 最终版 SphereTTC × 11 backbone；B = SphereDyn + SphereTTC） |
|---|---|---|
| `scripts/run_ttc.py` | 所有在线 TTC 方法的统一入口（raw / bias / affine / geo / spectral / st_ttc / sphere_ttc / sphere_ttc_v2–v7）；负责 causal memory、warmup、RMSE/ACC 指标 | **A**；B 中用于算指标 |
| **`src/spherettc.py`** | **SphereTTC 完整模块（单文件）**：Part 1 延迟核验记忆（`eligible_from_buffer`）、Part 2 球谐仿射校准器 `SphereTTCCalibrator`、Part 3 在线插件 `SphereTTC`、Part 4 多提供方约束组合（v22） | **A**（Part 1–3）、**B**（Part 4） |
| `src/ttc/metrics.py` | 纬度加权指标 | A、B |
| `src/utils/` | 配置、设备、时间、climatology 下标、zarr/xarray 工具 | A、B |
| `src/data_sources/s2s_daily.py` | 读取 1.5° UTC 日均 truth（`--truth-root`）与 SphereDyn 输入 | A、B |
| `src/data_sources/*_official.py`、`daily_mean.py` | 5 个官方模型的推理 adapter，生成 2017–2020 日均预报缓存 | A（官方 backbone）；B（参考预报） |
| `src/experiments/protocol.py` | common-49 变量与气压层定义 | A、B |
| `configs/unified_11model_ttc.yaml` | `run_ttc.py` 的配置 | **A** |
| `configs/official_daily_common49_2018_2019.yaml` | 官方模型日均缓存的生成配置 | A、B |
| `configs/unified_11model_protocol.yaml` | 11 模型统一协议 | A、B |
| `src/ttc/comparators/` | 对照方法：`geo.py`（geo_ttc）、`spectral.py`（zonal_spectral_ttc）、`st_ttc.py`（NeurIPS 2025 ST-TTC / SDCalibrator 的天气缓存版） | 否（论文对照需要） |
| `src/ttc/legacy/` | SphereTTC v2–v7：分频带、多时间尺度专家、季节残差专家等（见 §3）；proposal 中的 multi-timescale / seasonal experts 对应这里，但没有任何冻结结果使用 | 否 |
| **`src/spheredyn.py`** | **SphereDyn 完整模型（单文件）**：球面局部块、截断球谐谱动力学、semi-Lagrangian 输运、谐波气候趋势、观测时刻通路、多尺度残差，类 `SphereDyn` | **B** |
| `src/baselines/models.py` | 4 个本地 baseline（ConvLSTM / FNO / Transformer / ViT）与 `build_model`（`spheredyn_v9_multiscale` → `src/spheredyn.py`） | A（本地 backbone）；B |
| `src/baselines/external_models.py` | CirT / ClimODE 外部实现的封装 | A（本地 backbone） |
| `scripts/spheredyn/` | SphereDyn 推理、SphereTTC-v22 的 2017 统计累积与应用（数学在 `src/spherettc.py` Part 4）、指标、门槛判定（含两个 4-GPU 启动脚本） | **B** |
| `scripts/tables/` | 从逐 initialization 指标汇总 CSV，并渲染 49 张中文表 | 用于出表 |
| `scripts/preprocessing/build_daily_dayofyear_climatology.py` | 生成 1979–2016 逐日 ACC climatology | A、B（ACC） |
| `artifacts/graphcast_spherettc_20260804/schedule/tools/` | GraphCast `l64_s08` 的专用工具：2017 搜索、冻结、2018–2019 回看、2019/2020 GraphCast 推理、2020 truth、prospective、bootstrap、报告 | A（GraphCast 行与 2020 稳健性） |
| `artifacts/graphcast_spherettc_20260804/parameter_only/tools/` | parameter-only 负结果实验的工具 | 否（负结果） |
| `tools/build_results_summary.py` | 生成 `results/` 下的摘要（本次整理新增） | 摘要 |
| `verify_migration.py` | 只读一键验证（本次整理更新） | 验证 |
| `tests/` | 单元测试（合成数据）；两个依赖原始 S2S 数据的 loader 测试在缺数据时自动 skip | — |
| `archive/` | 整理前快照、原 README、原依赖文件、只能在原服务器全量归档里运行的验证器、整理日志；`pre_consolidation_20261003/` 保存合并前的 SphereTTC / SphereDyn 原文件 | 历史 |

## 2. 调用链

### SOTA A：最终版 SphereTTC（以 GraphCast `l64_s08` 为例；其余 10 个 backbone 用 `params/<model>.json`，调用方式相同）

```
artifacts/graphcast_spherettc_20260804/schedule/tools/
  search_schedule.py          # 2017：56 个候选 × search/holdout 两段 → VALIDATION_CANDIDATES_2017.csv
  analyze_validation.py
  freeze_primary.py           # 选出 l64_s08 → FROZEN_PRIMARY_GLOBAL_2017.json
  run_retrospective_primary.py   # 2018–2019（warmup：前一年缓存）
  run_graphcast_holdout.py    # 官方 GraphCast-small 生成 2019 warmup + 2020 预报（JAX，A100）
  prepare_truth_holdout.py    # WeatherBench2 ERA5 daily → 2020-01-01..2021-01-10 truth
  run_prospective_holdout.py  # 2020 prospective
  bootstrap_prospective.py    # 14 天 block bootstrap
  finalize_report.py / integrity.py
        │  每一步都以子进程调用
        ▼
scripts/run_ttc.py --method sphere_ttc --params-json configs/generated/l64_s08.json
        ├── src/spherettc.py       eligible_from_buffer（per-lead delayed memory，注意 R1）
        │                          SphereTTCCalibrator（SHT 截断 → 仿射拟合 → chronological gate → taper → 逆 SHT）
        ├── src/ttc/metrics.py     cos(lat) 加权 RMSE / ACC
        └── src/data_sources/s2s_daily.py  truth
```

2018–2019 原始命令可以在 `schedule/logs/retrospective_l64_s08_2018.log` 第 3 行看到。

### SOTA B：SphereDyn + SphereTTC

```
scripts/spheredyn/run_main_prediction_v2_4gpu.sh
  ├── scripts/spheredyn/infer_spatiotemporal_checkpoint.py   # SphereDyn-v9 推理 2017 / 2019（2018 复用筛选阶段缓存）
  │       └── src/baselines/__init__.py::build_model → src/spheredyn.py::SphereDyn
  ├── scripts/spheredyn/run_main_spherettc_v2.py             # SphereTTC-v22：2017 拟合约束组合权重并冻结，作用于 2018–2019
  │       └── src/spherettc.py Part 4（经 evaluate_goal_reference_ensemble.py 转出）   # 候选权重、岭回归、空间 bias 收缩
  └── scripts/spheredyn/assess_main_prediction_goal.py       # 5% 主门槛 → MAIN_GOAL_GATE.json
scripts/spheredyn/compute_spheredyn_main_rmse_acc_4gpu.sh
  ├── scripts/run_ttc.py --method raw                          # 对 Raw / Final 缓存算 RMSE、ACC
  ├── scripts/spheredyn/merge_metric_npz.py
  └── scripts/spheredyn/build_spheredyn_main_rmse_acc_table.py # 490 行 CSV
```

## 3. 版本谱系（为什么文件里有这么多 v）

### SphereTTC

| 版本 | 位置 | 内容 | 状态 |
|---|---|---|---|
| SphereTTC（代码里没有版本号） | `src/spherettc.py` Part 1–3（原 `src/ttc/sphere.py` + `memory.py`） | 截断球谐 + 鲁棒加权仿射 + chronological gate + taper | **最终版（单 backbone 模式）**：SOTA A（11 个 backbone，含 GraphCast `l64_s08`） |
| v2 | `src/ttc/legacy/sphere_v2.py` | 分频带、分分量风险控制 | 无冻结结果 |
| v3 | `legacy/sphere_v3.py` | 频带共享 + validation guard | 无冻结结果 |
| v4 | `legacy/sphere_v4.py` | 短/长记忆专家混合（proposal：multi-timescale experts） | 无冻结结果 |
| v5 | `legacy/sphere_v5.py` | 三时间尺度 + 受保护的 no-op | 无冻结结果 |
| v6 | `legacy/sphere_v6.py` | 四专家 + 保守 bias anchor + block-robust | 无冻结结果 |
| v7 | `legacy/sphere_v7.py` | 加季节残差 memory（proposal：seasonal experts） | 无冻结结果 |
| v8–v21 | 已在 2026-08-03 清理时移除（见 `artifacts/spheredyn_spherettc_open_goal_20260801/CLEANUP_RECORD.md`） | — | — |
| v22（constrained reference anchor） | `src/spherettc.py` Part 4（原 `evaluate_goal_reference_ensemble.py` 的数学部分），由 `scripts/spheredyn/run_main_spherettc_v2.py` 调用 | **最终版（多提供方约束组合模式）**：backbone 权重 ≥ 0.5，与 5 个参考预报按 lead × 变量组合，加全局与空间 bias；2017 拟合并冻结，不含球谐分量 | **SOTA B**（SphereDyn + SphereTTC）；`EXPERIMENT_STATE_V2.json` 记为最佳 SphereTTC 开发候选 |

### SphereDyn

`models.py::SphereDynForecast`（v1）→ `spheredyn_v2`（semi-Lagrangian 输运 + 谐波气候趋势）→ `v4`（球面历史动力学）→ `v6`（递归日流）→ `v7` → `v9`（多尺度球面残差，最终 checkpoint `spheredyn_v9_multiscale.pt`，约 400 万参数，seed 44）。v8 的递归路径与观测时刻解码器按 v9 docstring 原样保留在 v9 中，单独的 v8 文件不在包内。

**2026-10-03 起，最终模型（v9 及其依赖的 v2/v4/v6/v7 组件）合并为单文件 `src/spheredyn.py`**，类名改为可读的名字（`SphericalDepthwiseBlock` → `SphericalLocalBlock`，`StableSphericalBandDynamics` → `SphericalSpectralDynamics`，`PeriodicSemiLagrangianTransport` → `SemiLagrangianTransport`，`SphericalHistoryDynamics` → `SphericalHistoryResponse`，`SphereDynV9Forecast` → `SphereDyn`），参数名、随机初始化顺序与计算顺序不变；`tests/test_consolidation_equivalence.py` 验证与原文件逐位相同的输出。原文件保存在 `archive/pre_consolidation_20261003/src/baselines/`。**没有训练脚本**，只能用现有 checkpoint 推理（2026-10-02 已验证 checkpoint 能以 `strict=True` 加载并完成前向，见 `tests/test_spheredyn_checkpoint.py`）。checkpoint 的 `args` 记录了最后一阶段的训练配方：从 v8 dual-path checkpoint 热启动，8 epochs，batch 2，lr 5e-5，weight decay 1e-4，loss 为 common-49 macro RMSE，6,000 个 fit 样本，355 个 validation 样本，seed 44。v8 checkpoint 和更早的训练链不在本包里。

## 4. 代码溯源校验

两份日志记录了全部改动：`archive/provenance/RESTRUCTURE_LOG_20261002.json`（目录整理）和 `archive/provenance/CONSOLIDATION_LOG_20261003.json`（SphereTTC / SphereDyn 合并为单文件）。`verify_migration.py` 会：

1. 依次重放两份日志的移动、反推所有字符串替换，把每个文件还原为 2026-08-06 迁移包的字节，与 `archive/provenance/MIGRATION_MANIFEST_20260806.sha256` 逐一比对；
2. 再去掉迁移时加的 `sys.path` 引导，与实验运行时记录的源码哈希（`artifacts/graphcast_spherettc_20260804/schedule/integrity/source_before.json`）比对；
3. 检查并入 `src/spherettc.py` 的 14 个函数 / 类 / 常量与原文件在 AST 上完全相同（只允许 docstring 和注释不同）。

SphereDyn 合并时改了类名并把继承链展开成一个类，无法做 AST 比对，因此由 `tests/test_consolidation_equivalence.py`（需要 torch）验证：随机初始化逐位相同，小模型随机参数和完整 checkpoint 的前向输出都逐位相同。

第 2 步的结果：`src/` 下全部文件（含 SphereTTC 核心与 SphereDyn 全部版本），以及 `scripts/run_ttc.py`、`build_daily_dayofyear_climatology.py`、`infer_spatiotemporal_checkpoint.py`、`run_main_spherettc_v2.py` 等，都与 2026-08-04 记录的运行时版本一致。例外有两个：`scripts/spheredyn/evaluate_goal_reference_ensemble.py`（SOTA B 中 SphereTTC-v22 的权重选择逻辑）有路径以外的差异，无法逐字节对齐；`verify_main_results_reproducibility.py` 也有差异，但它只是验证器。另外，运行时哈希记录于 2026-08-04，而 SOTA B 是 08-03 跑的，所以即使对得上，也只能证明与 08-04 的版本一致。SOTA B 的数值可复现性目前依靠冻结的 `FITTED_WEIGHTS_2017.npz` 和 490 行指标，而不是代码哈希。

## 5. 时间线

| 日期（2026） | 事件 | 产物 |
|---|---|---|
| 07-23 | 本地 baseline 的 2017 验证缓存；SphereTTC 参数精调 | 原服务器目录 `ttc_publication_local_*`、`ttc_publication_spherettc_*`（不在本包） |
| 07-26 | 11 baseline publication SphereTTC 冻结（2017 选参，2018–2019 测试） | `artifacts/ttc_publication_corrected_20260726/` |
| 08-01 → 08-03 | SphereDyn 开放目标实验；协议 v1 → v2；v9 + v22 主结果 | `artifacts/spheredyn_spherettc_open_goal_20260801/` |
| 08-03 | 工程清理，v2–v21 等历史产物移出 | `CLEANUP_RECORD.md` |
| 08-04 | GraphCast parameter-only（FAIL）→ schedule 搜索 → `l64_s08` → 2020 prospective PASS | `artifacts/graphcast_spherettc_20260804/` |
| 08-06 | 轻量迁移包（约 55 MiB） | `archive/old_docs/README_ZH_lightweight_migration_20260806.md` |
| 10-02 | 原地整理目录结构 | `archive/provenance/RESTRUCTURE_LOG_20261002.json` |
| 10-03 | SphereTTC → `src/spherettc.py`，SphereDyn → `src/spheredyn.py`（单文件，验证等价） | `archive/provenance/CONSOLIDATION_LOG_20261003.json` |
