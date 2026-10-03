# SOON 最新完整实验结果（截至 2026-08-04）

## 结论

原 49 表已达到 SphereDyn 主目标并保持为不可变历史基准。后续 GraphCast + SphereTTC 确有实验进展：推荐将单一全局 `l64_s08` 作为最新 GraphCast 主结果；family/lead schedule 只作为更灵活的次要分析。

| 实验 | 时期 | 平均单元 RMSE 改善 | RMSE 退化 | macro ACC 变化 | ACC 退化 | 判定 |
|---|---|---:|---:|---:|---:|---|
| 原 49 表 GraphCast + publication SphereTTC | 2018–2019 | 2.4346% | 2/245 | -0.00205418 | 94/245 | 历史基准 |
| parameter-only `l24_strength050` | 2018–2019 | 1.5482% | 1/245 | -0.00001117 | 55/245 | FAIL |
| 全局 `l64_s08` | 2018–2019 | 3.3834% | 2/245 | +0.00154237 | 21/245 | 同口径改善 |
| 全局 `l64_s08` | 2020 prospective | 3.5734% | 0/245 | +0.00256941 | 10/245 | PASS |
| family/lead schedule | 2020 prospective | 3.6758% | 0/245 | +0.00297449 | 1/245 | PASS（次要） |

全局 `l64_s08` 相比原 49 表中的 GraphCast + SphereTTC，把同一 2018–2019 测试上的平均 RMSE 改善提高 0.9488 个百分点，并把 macro ACC 从负变化改为正变化；ACC 退化单元由 94 降至 21。

## 权威结果入口

- 原冻结 49 表：`../spheredyn_spherettc_open_goal_20260801/MAIN_RESULTS_49_VARIABLES_ZH.md`
- 更新 GraphCast 行后的最新 49 表：`LATEST_RESULTS_49_VARIABLES_ZH.md`
- 机器可读最新 5,880 行：`LATEST_RESULTS.csv`
- 2020 prospective 结果：`schedule/PROSPECTIVE_2020_REPORT.md`
- 完整数值对比：`COMPARISON_WITH_FROZEN_49_TABLES.json`

## 解释边界

- SphereDyn 主结果仍只有 seed 44，且 SphereTTC-v22 使用外部参考预测。
- `l64_s08` 是一个全局参数集，2017 冻结后才用于 2018–2019 与 2020；2020 是新增的独立 prospective 证据。
- family/lead schedule 自由度更高，因此不替代全局参数作为 headline。
- 2020 只新增 GraphCast 对照，不能把它与其他模型的 2018–2019 数值直接排名。
