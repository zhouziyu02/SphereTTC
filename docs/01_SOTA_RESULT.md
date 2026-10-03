# 01 · SOTA 结果：最终版 SphereTTC，与 SphereDyn + SphereTTC

> 机器可读汇总：`results/SOTA_SUMMARY.json`。表格：`results/SPHERETTC_11_BACKBONES.md`、`results/SPHEREDYN_SPHERETTC.md`、`results/MACRO_SUMMARY_2018_2019.csv`、`results/HEADLINE_{120H,240H}_LATEST.md`。
> 这些文件都由 `tools/build_results_summary.py` 从冻结的 `artifacts/` 确定性生成，`verify_migration.py` 会逐字节核对。

统一评估口径：2018-01-01 至 2019-12-31，730 个逐日 initialization；common-49 变量（45 个高空 + 4 个地面），24/72/120/168/240h 五个时效；cos(lat) 加权 RMSE 与 ACC（climatology 只用 1979–2016）。主指标 macro nRMSE 是 49 变量 × 5 时效上“RMSE / 1979–2016 变量标准差”的等权平均。所有校准参数只用 2017 年选择，冻结后再评估 2018–2019。

## 0. 一览

两个 SOTA 互补：A 证明 SphereTTC 作为即插即用的 calibrator 能有效提升现有 baseline；B 证明球面机理对这个任务有用，同时 SphereDyn 是同协议下的 SOTA baseline。论文叙事见 `05_PAPER_STORYLINE.md`。

| SOTA | 设置 | 关键结果（2018–2019） |
|---|---|---|
| **A. 最终版 SphereTTC（即插即用）** | 冻结 11 个已有 backbone（6 个本地训练 + 5 个官方发布模型），每个 backbone 外挂 SphereTTC | **11/11 个 backbone 的 macro nRMSE 都下降**，平均下降 4.20%（中位 1.34%，范围 0.10%–19.82%）；5 个时效上平均都下降（24h 6.28% … 240h 4.54%）；9/11 个 backbone 的 macro ACC 上升 |
| **B. SphereDyn + SphereTTC** | SphereDyn-v9（seed 44）+ SphereTTC-v22 | macro nRMSE **0.3809 → 0.2846（−25.28%）**；**245/245 个单元 RMSE 和 ACC 全部提升**；macro ACC 0.8080，**全表 24 行第一**；168h、240h 两个时效的 nRMSE 和 ACC 都是**全表第一** |

## 1. “最终版 SphereTTC”指的是什么

仓库里的 SphereTTC 有很多版本号，最终产出 SOTA 数字的是下面两种运行模式。它们正好对应 proposal 中 SphereTTC 的两部分：

| 模式 | 代码 | 产出的结果 | 对应 proposal 中的描述 |
|---|---|---|---|
| **单 backbone 球面校准**（代码里没有版本号，就叫 SphereTTC） | `src/spherettc.py`（Part 1–3：`eligible_from_buffer`、`SphereTTCCalibrator`、在线插件 `SphereTTC`），实验入口 `scripts/run_ttc.py --method sphere_ttc` | **A**：11 个 backbone 的全部结果；GraphCast 的更新参数 `l64_s08` 与 2020 prospective | truncated spherical-harmonic space；conservative affine corrections from previously verified pairs；recency + robust weighting；chronological validation；risk-aware gate 与 no-op |
| **多提供方约束组合**（SphereTTC-v22，“constrained reference anchor”） | `src/spherettc.py` Part 4（`fit_constrained_combination` / `apply_constrained_combination`），由 `scripts/spheredyn/run_main_spherettc_v2.py` 调用 | **B**：SphereDyn + SphereTTC | “When multiple forecast providers are available, SphereTTC can further learn lead- and variable-specific constrained combinations with trusted reference systems while preserving the contribution of the target backbone” |

- `EXPERIMENT_STATE_V2.json` 把 v22 记为最好的 SphereTTC 开发候选（`best_spherettc_development_candidate`）。
- v2–v7（`src/ttc/legacy/`：分频带、多时间尺度专家、受保护 no-op、bias anchor、季节残差专家）在本包里**没有任何冻结结果**。v8–v21 的代码和产物在 2026-08-03 清理时已经移出（`CLEANUP_RECORD.md`）。proposal 里的 “multi-timescale and seasonal calibration experts” 对应的就是这些版本，但最终数字没有用到它们（见 `04_AUDIT_AND_RISKS.md` R11）。

## 2. 结果 A：SphereTTC 在 11 个已有模型上的一致提升

### 2.1 方法与参数

- `SphereTTCCalibrator` 的流程：截断球谐变换（SHT）→ 在 `lmax × mmax` 系数空间逐变量拟合“岭回归趋向恒等变换”的仿射校正 → recency（半衰期）与 robust 加权 → 按时间顺序切出 holdout，估计保守 gate（置信下界 × 最优混合系数 × strength）→ 按 ℓ 做 taper → 逆 SHT。每个 lead 只使用已经核验过的历史“预报–真值”对，backbone 全程冻结。
- 每个 backbone 的参数都在 2017 年选择（init 120 以后计分，目标为“带负迁移惩罚的 macro 相对 RMSE 增益”）并冻结：`artifacts/ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json` 与 `params/*.json`。
- GraphCast 使用后来同样只用 2017 选出的全局参数 `l64_s08`（lmax 64，strength 0.8，冻结于 2026-08-04；见 `artifacts/graphcast_spherettc_20260804/schedule/`）。原 publication 参数（lmax 24）在 GraphCast 上的 macro nRMSE 变化是 −3.11%。

### 2.2 2018–2019 主表

| Backbone | Raw nRMSE | +SphereTTC | 增益 | RMSE 改善 / 变差单元 | macro ACC Raw → +SphereTTC |
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

（表中“增益”指 macro nRMSE 的相对变化，负号表示误差下降。）

- 按时效的平均变化：24h −6.28%（11/11 改善），72h −3.95%（11/11），120h −2.99%（11/11），168h −3.34%（10/11），240h −4.54%（10/11）。
- macro ACC 在 9/11 个 backbone 上升；OneForecast（−0.00038）和 Pangu（−0.00069）略降。
- GraphCast + SphereTTC 的 macro nRMSE 0.2784 是 2018–2019 全表 24 行中最低的。

### 2.3 稳健性：GraphCast 的 2020 独立 prospective holdout

`l64_s08` 冻结后，用官方 GraphCast-small 1° checkpoint 重新生成 2020 全年预报（2019 年最后 206 天做因果 warmup）：macro nRMSE 0.2946 → 0.2811；0/245 个单元 RMSE 变差；平均单元 RMSE 改善 3.57%，14 天 block bootstrap 95% 区间 [3.42%, 3.73%]；macro ACC +0.00257；预注册门槛 PASS。逐时效结果见 `schedule/PROSPECTIVE_2020_REPORT.md`。同一 backbone 上的参数对照（publication、parameter-only 负结果、family/lead schedule）可以作为消融放进附录，见 `MAIN_EXPERIMENT_RECOMMENDATION.md`。

## 3. 结果 B：SphereDyn 与 SphereDyn + SphereTTC

### 3.1 SphereDyn backbone

- 模型：SphereDyn-v9 multiscale，seed 44，约 400 万参数。checkpoint 为 `artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt`（SHA-256 `121f9b90…f453`）。
- 结构对应 proposal：v2 的 semi-Lagrangian 输运与谐波气候趋势、v4/v6 的截断球谐谱动力学与递归日流、球面局部块、v8 的直接观测时刻通路，以及 v9 新增的三尺度球面残差。完整模型在单文件 `src/spheredyn.py`（类 `SphereDyn`）。
- Raw SphereDyn 的 macro nRMSE 是 0.3809，比全部 6 个同协议训练的 baseline 都好（最好的 FNO 为 0.4427，低 14%）；而且在 5 个 lead 上，nRMSE 和 ACC 两个指标都赢，所以可以称为同协议下的 SOTA baseline。它也好于官方发布的 FourCastNetV2（0.4500），但不如其余 4 个官方模型。在 240h，它的 nRMSE 是全部 12 个 Raw 模型中最低的（0.4801，GraphCast 为 0.5173），不过同一时效的 ACC 只排第 6，这一项优势部分来自预报更平滑。

### 3.2 SphereDyn + SphereTTC

| 时效 | SphereDyn Raw | SphereDyn + SphereTTC | 增益 | ACC Raw → +TTC | GraphCast Raw | +TTC 在全表的 nRMSE / ACC 排名 |
|---|---:|---:|---:|---:|---:|---:|
| 24h | 0.1870 | 0.1274 | −31.87% | 0.9320 → 0.9683 | 0.1133 | 7 / 7 |
| 72h | 0.3595 | 0.2271 | −36.84% | 0.7242 → 0.9051 | 0.1788 | 8 / 8 |
| 120h | 0.4226 | 0.2922 | −30.86% | 0.5825 → 0.8375 | 0.2706 | 7 / 7 |
| 168h | 0.4553 | **0.3532** | −22.42% | 0.4781 → **0.7458** | 0.3785 | **1 / 1** |
| 240h | 0.4801 | **0.4231** | −11.87% | 0.3694 → **0.5833** | 0.5173 | **1 / 1** |
| 全部 | 0.3809 | **0.2846** | **−25.28%** | 0.6172 → **0.8080** | 0.2917 | 2 / **1** |

- 冻结的主门槛（`main_prediction_v2_seed44/MAIN_GOAL_GATE.json`）：要求相对同次 Raw SphereDyn 至少提升 5%，实际 25.2829%，状态 `MAIN_PREDICTION_GOAL_ACHIEVED`。
- 245/245 个单元 RMSE 下降、ACC 上升。
- macro nRMSE 0.2846 低于所有 Raw 模型（含 GraphCast 0.2917、Pangu 0.2983），在 24 行里仅次于 GraphCast + SphereTTC（0.2784）；macro ACC 全表第一。
- v22 的权重形式：每个（lead, 变量）上 SphereDyn 权重固定为 0.5 或 0.75（或保持 Raw），其余权重分给 5 个参考预报，再加全局与空间 bias；2017 年拟合并冻结。拟合权重为 `main_prediction_v2_seed44/final/FITTED_WEIGHTS_2017.npz`。

## 4. Proposal 论断与证据对照

| Proposal 中的说法 | 本仓库的证据 | 状态 |
|---|---|---|
| SphereTTC consistently improves forecast accuracy across variables, horizons and backbones | 11/11 个 backbone macro nRMSE 下降；5 个时效平均增益全为正；单元级别 CirT 有 93/245、ViT 有 54/245 个单元变差 | 宏观层面成立；单元层面需如实描述 |
| … and anomaly correlation | 9/11 个 backbone macro ACC 上升，OneForecast、Pangu 略降 | 大体成立，要写明例外 |
| Truncated SH、recency/robust weighting、chronological validation、risk-aware gate + no-op | `src/spherettc.py` 全部实现，并用于结果 A | 成立 |
| Multi-timescale and seasonal calibration experts | 代码在 `src/ttc/legacy/sphere_v4–v7.py`，但没有任何冻结结果使用 | **缺证据**（R11） |
| Constrained combinations with trusted reference systems | v22，用于结果 B | 成立，但增益来源需要拆分（R2） |
| SphereDyn：daily flow、spherical local + spectral dynamics、semi-Lagrangian、harmonic climate tendency、observation-time pathway、multi-resolution residual | `src/spheredyn.py` | 成立（单 seed） |
| Integration with SphereDyn gives an effective end-to-end spherical system | 结果 B：−25.28%，全表最佳 macro ACC，168/240h 第一 | 成立，但有 R2/R3 的注意事项 |
| Experiments on ERA5 | ERA5 1.5° UTC 日均值（S2S 切片 + WeatherBench2） | 成立 |

## 5. 证据文件

| 内容 | 路径 |
|---|---|
| 11 backbone 冻结参数 | `artifacts/ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json`、`params/*.json` |
| 11 backbone × 2 方法 × 49 变量 × 5 时效（原版） | `artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv`（含分年份版本 `MAIN_RESULTS_BY_YEAR.csv`） |
| 11 backbone 宏观汇总（原版） | `artifacts/spheredyn_spherettc_open_goal_20260801/UNIFIED_11_SPHERETTC_MACRO_SUMMARY.json` |
| 最新 5,880 行 / 49 张表 | `artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`、`LATEST_RESULTS_49_VARIABLES_ZH.md` |
| GraphCast `l64_s08` 选择与 2020 prospective | `artifacts/graphcast_spherettc_20260804/schedule/`（`FROZEN_PRIMARY_GLOBAL_2017.json`、`PROSPECTIVE_2020_*`） |
| SphereDyn 协议、门槛与汇总 | `artifacts/spheredyn_spherettc_open_goal_20260801/PROTOCOL_V2.json`、`main_prediction_v2_seed44/MAIN_GOAL_GATE.json`、`EXPERIMENT_SUMMARY_ZH.md` |
| SphereDyn 490 行 RMSE/ACC | `artifacts/spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/paper_metrics/SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv` |
| 权重 | `spheredyn_v9_multiscale.pt`（16,302,127 字节）、`FITTED_WEIGHTS_2017.npz`（35,588,308 字节） |

## 6. 写论文时的建议

可以直接说的（前提是先修复并重跑 R1）：

- SphereTTC 不改动 backbone，只用已核验的历史样本，在 11 个结构各异的 backbone 上全部降低了 macro nRMSE，平均 4.20%，每个时效的平均增益都为正。
- SphereTTC 在最强的 GraphCast 上也有效（−4.57%），并在独立的 2020 年 prospective holdout 上 0/245 个单元变差。
- SphereDyn 优于所有同条件训练的 baseline，并在 240h 优于所有 Raw 模型；SphereDyn + SphereTTC 取得全表最佳 macro ACC，以及 168h、240h 两个时效的最佳 nRMSE 与 ACC。

需要措辞小心的：

- “consistently improves ACC”：要写明 OneForecast 和 Pangu 的 macro ACC 略降。
- SphereDyn + SphereTTC 的 25.28% 包含了与参考预报的约束组合，正文里要同时给出拆分（SphereDyn + 单 backbone SphereTTC、只用参考预报的组合）。补实验前不要把全部增益归因于 SphereDyn 本身。
- FourCastNetV2（−19.82%）和 FuXi（−9.80%）的大增益主要来自比湿（R5），建议同时报告不含 q 的指标。
- 2020 的 GraphCast 数字不要和其他模型的 2018–2019 数字混在同一张表里排名。
