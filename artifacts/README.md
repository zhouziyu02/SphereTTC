# artifacts/ — SphereTTC 冻结实验证据

当前结果仅包含 11 个冻结 backbone 的 Raw 与 SphereTTC。清理只筛除另一项目的记录并重建展示文件；保留的逐单元 RMSE/ACC 字符串、原始 publication CSV、GraphCast 更新 CSV 和标准化统计均未修改。`verify_migration.py` 固定关键文件 SHA-256，并独立重建结果与表格。

| 目录 | 内容 |
|---|---|
| `ttc_publication_corrected_20260726/` | 11-backbone publication 参数、5,390 行结果、稿件表与 climatology 元数据 |
| `graphcast_spherettc_20260804/` | GraphCast `l64_s08` 更新、2017 选参、2018–2019 回看、2020 prospective 和 parameter-only 负结果 |
| `shared/` | 1979–2016 的 54 变量标准化统计；汇总 nRMSE 时使用其中的 common-49 变量 |
| `provenance/` | 只读历史完整性记录；其中旧路径不是本仓库运行依赖 |

最新入口：

- `graphcast_spherettc_20260804/LATEST_RESULTS.csv`：5,390 行，11 backbones × 2 methods × 49 variables × 5 leads。
- `graphcast_spherettc_20260804/LATEST_RESULTS_49_VARIABLES_ZH.md`：49 张表，每张 22 行。
- `graphcast_spherettc_20260804/schedule/PROSPECTIVE_2020_REPORT.md`：独立年份 GraphCast 证据。
- `ttc_publication_corrected_20260726/results/MAIN_RESULTS_49_VARIABLES_ZH.md`：原 publication 参数表。

从仓库根目录运行 `PYTHONDONTWRITEBYTECODE=1 python verify_migration.py` 做只读检查；`python scripts/tables/build_latest_results.py --check` 与 `python tools/build_results_summary.py --check` 分别检查最新全表与摘要。当前轻量仓库不包含预测缓存、原始天气数据、逐 initialization 指标或 checkpoint；不能将表格重建等同于重新预测或重新 bootstrap。

校验范围为受清单管理的分发包；用户本地 `paper/`、`.venv/` 和 `runs/` 不在分发包清单或轻量数据策略范围内。
