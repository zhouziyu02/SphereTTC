# artifacts/ —— 冻结实验证据（只读）

这里的文件是实验当时冻结的结果、配置、日志与审计，整理时**没有修改任何一个字节**。历史日志、profile 里的 `/mnt/bn/...` 绝对路径只是原服务器 provenance。

| 目录 | 日期 | 内容 | 状态 |
|---|---|---|---|
| `graphcast_spherettc_20260804/` | 08-04 | GraphCast 上 SphereTTC 的参数更新：`schedule/`（56 候选 → **`l64_s08`**，2018–2019 回看，2020 prospective）和 `parameter_only/`（负结果）；更新后的 5,880 行结果与 49 张表 | SOTA A 的 GraphCast 一行 + 稳健性；**最新全表** |
| `spheredyn_spherettc_open_goal_20260801/` | 08-01 → 08-03 | SphereDyn-v9 + SphereTTC-v22 主预测、协议 v2、原版 49 张表、两份权重 | **SOTA B**（注意 R2/R3） |
| `ttc_publication_corrected_20260726/` | 07-26 | 11 个 backbone 的 SphereTTC 冻结参数与 5,390 行结果、120h 稿件表、climatology 元数据 | **SOTA A**（10 个 backbone 直接来自这里） |

最常用的几个文件：

- 最新结果（机器可读）：`graphcast_spherettc_20260804/LATEST_RESULTS.csv`
- 最新 49 张表：`graphcast_spherettc_20260804/LATEST_RESULTS_49_VARIABLES_ZH.md`
- SOTA A：`ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json`、`results/MAIN_RESULTS.csv`；GraphCast 更新见 `graphcast_spherettc_20260804/schedule/MAIN_EXPERIMENT_RECOMMENDATION.md`
- SOTA B：`spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/MAIN_GOAL_GATE.json`、`EXPERIMENT_SUMMARY_ZH.md`
- 2020 prospective：`graphcast_spherettc_20260804/schedule/PROSPECTIVE_2020_REPORT.md`
- 原版 49 张表：`spheredyn_spherettc_open_goal_20260801/MAIN_RESULTS_49_VARIABLES_ZH.md`

各子目录 README 里提到的 `scripts/verify_*_reproducibility.py` 已移到 `archive/full_archive_verifiers/`；`scripts/build_latest_results.py` 等出表脚本现在位于 `scripts/tables/`。本仓库的一键验证请使用根目录的 `verify_migration.py`。
