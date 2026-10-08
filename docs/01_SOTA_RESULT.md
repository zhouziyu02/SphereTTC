# 01 · SphereTTC：11 个 backbone 的冻结结果

> 机器可读汇总：`results/SOTA_SUMMARY.json`（保留既有文件名）。主表：`results/SPHERETTC_11_BACKBONES.md`、`results/MACRO_SUMMARY_2018_2019.csv`、`results/PAPER_TABLE1_NRMSE_BY_LEAD.md`、`results/PAPER_TABLE2_ACC_BY_LEAD.md`。
> 这些摘要由 `tools/build_results_summary.py` 从冻结证据确定性生成。当前范围为 **11 个 backbone、22 个 Raw / +SphereTTC 配置、5,390 个逐变量逐时效指标行**。

**证据边界：本页记录现有冻结结果，不等于完成投稿前验证。** 五个官方 backbone（包括 GraphCast 的 2020 结果）存在日均 truth 进入 memory 过早的问题，必须修复并重跑；完整风险见 `04_AUDIT_AND_RISKS.md`。此次代码库整理不改变历史算法和数值。

## 1. 评估口径与主结论

2018-01-01 至 2019-12-31，730 个逐日 initialization；common-49 变量（45 个高空 + 4 个地面），24/72/120/168/240 h 五个时效；cos(lat) 加权 RMSE 与 ACC（climatology 只用 1979–2016）。macro nRMSE 是 49 变量 × 5 时效上 RMSE / 1979–2016 变量标准差的等权平均。参数候选的数值选择使用 2017；GraphCast 搜索空间调整前已经查看过 2018–2019 结果，不能将其开发历史描述为完全盲测。

冻结结果中，11/11 个 backbone 的 macro nRMSE 下降，平均下降 4.20%（中位 1.34%，范围 0.10%–19.82%）；五个时效的平均变化均为改善。macro ACC 在 9/11 个 backbone 上升，OneForecast、Pangu 略降。宏观改善不代表每个变量–时效单元都改善，也没有证明相对于其他 TTC 方法达到 SOTA。

## 2. 方法与参数

最终方法只有一种：**单 backbone 球面校准**。`src/spherettc.py` 包含 delayed memory、`SphereTTCCalibrator` 和在线插件 `SphereTTC`；实验入口为 `scripts/run_ttc.py --method sphere_ttc`。

流程为截断球谐变换 → 每个变量和时效的保守谱空间仿射拟合 → recency 与 robust 加权 → chronological holdout gate → degree taper → 逆变换并加回原预报。岭回归向恒等变换收缩、scale 偏离受限；gate 为零时不修改预报。backbone 全程冻结，不做梯度训练，但校准器会在线拟合校准系数。

- publication 参数在 2017 选择（init 120 以后计分，目标为带负迁移惩罚的 macro 相对 RMSE 增益），冻结记录为 `artifacts/ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json`、`params/*.json`。
- GraphCast 主表使用之后同样在 2017 选择的全局参数 `l64_s08`（lmax 64、strength 0.8），见 `artifacts/graphcast_spherettc_20260804/schedule/FROZEN_PRIMARY_GLOBAL_2017.json`。原 publication 参数 lmax 24 的 GraphCast macro nRMSE 变化为 −3.11%。两套选择协议的差异须披露（R6）。
- `src/ttc/legacy/` 的 v2–v7 是未进入冻结主结果的实验变体；不能把多时间尺度或季节专家写成当前主结果使用的组件。

## 3. 2018–2019 主表

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
- GraphCast + SphereTTC 的 macro nRMSE 0.2784 是 2018–2019 的 22 个 Raw / +SphereTTC 配置中最低的。

## 4. 稳健性：GraphCast 的 2020 prospective holdout

`l64_s08` 冻结后，用官方 GraphCast-small 1° checkpoint 重新生成 2020 全年预报（2019 年最后 206 天做 warmup）：macro nRMSE 0.2946 → 0.2811；0/245 个单元 RMSE 变差；平均单元 RMSE 改善 3.57%，14 天 block bootstrap 95% 区间 [3.42%, 3.73%]；macro ACC +0.00257；冻结门槛 PASS。这是原协议下的结果；同样受日均真值可用时刻问题（R1）影响，尚不能当作已验证的严格因果部署结果。逐时效结果见 `artifacts/graphcast_spherettc_20260804/schedule/PROSPECTIVE_2020_REPORT.md`。同一 backbone 上的参数对照（publication、parameter-only 负结果、family/lead schedule）可以作为消融放进附录，见 `artifacts/graphcast_spherettc_20260804/MAIN_EXPERIMENT_RECOMMENDATION.md`。

## 5. 论断与证据对照

| 论断 | 现有证据与限制 |
|---|---|
| 对异构 backbone 的宏观误差有改善 | 11/11 macro nRMSE 下降；官方模型结果须在修复 R1 后重新确认 |
| 对所有变量、时效都改善 | 不成立；例如 CirT 有 93/245、ViT 有 54/245 个 RMSE 单元变差 |
| ACC 一致改善 | 不成立；9/11 macro ACC 上升，OneForecast、Pangu 略降 |
| 球谐、recency/robust 加权、chronological gate、no-op 是实际方法 | 已在 `src/spherettc.py` 实现并用于冻结结果；各组件独立贡献仍需消融 |
| 多时间尺度和季节专家有效 | 有实验代码，没有支持该论断的冻结结果 |
| 无需梯度训练、开销低 | 冻结 profile 记录约 0.07 s 校准增量开销；硬件与计时边界见 `results/EFFICIENCY.md` |
| 严格因果部署、领先其他校准方法、首次研究 | 当前证据不足；需修复 R1、补对照、核对相关文献 |

## 6. 证据文件

| 内容 | 路径 |
|---|---|
| 11 backbone 冻结参数 | `artifacts/ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json`、`params/*.json` |
| publication 5,390 行及分年结果 | `artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv`、`MAIN_RESULTS_BY_YEAR.csv` |
| 更新 GraphCast 后的 5,390 行 / 49 张变量表 | `artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`、`LATEST_RESULTS_49_VARIABLES_ZH.md` |
| normalization statistics | `artifacts/shared/s2s_daily_54var_stats.json` |
| GraphCast 选择与 2020 prospective | `artifacts/graphcast_spherettc_20260804/schedule/`（`FROZEN_PRIMARY_GLOBAL_2017.json`、`PROSPECTIVE_2020_*`） |
| GraphCast parameter-only 负结果 | `artifacts/graphcast_spherettc_20260804/parameter_only/` |

写论文时应同时报告 RMSE 和 ACC、改善与回退单元数、变量族结果，并核查 FourCastNetV2 / FuXi 的 RH→q 换算和不含 q 的指标。2020 GraphCast 不能与其余模型的 2018–2019 数字混在同一排名中。任何严格因果主结论都须以修复 R1 后的新运行结果为准。
