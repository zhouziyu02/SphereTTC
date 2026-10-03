# SOON 主实验最终迁移工程

这是截至 **2026-08-04** 的 SOON 主实验轻量冻结版，打包日期为 2026-08-06。它保留代码、配置、实验审计、冻结汇总数值、全部 49 张主表，以及最终 SphereDyn-v9 / SphereTTC-v22 的两份权重；不包含原始气象数据、逐 initialization 指标、预测缓存、其他模型 checkpoint 或分卷压缩包。当前目录约 55 MiB。

## 最新结论

原版 49 表已经达到 SphereDyn 主目标，继续作为不可变历史基准。2026-08-04 的 GraphCast + SphereTTC 后续实验存在明确进展，最新版 49 表已用冻结的全局参数 `l64_s08` 更新其中 245 个 GraphCast + SphereTTC 单元；其余 5,635 行保持原值。

| 实验 | 时期 | 平均单元 RMSE 改善 | macro ACC 变化 | RMSE / ACC 退化单元 | 状态 |
|---|---|---:|---:|---:|---|
| SphereDyn-v9 + SphereTTC-v22 | 2018–2019 | macro nRMSE `0.3808743 → 0.2845782`（25.2829%） | — | — | 主目标达到 |
| 原 49 表 GraphCast + publication SphereTTC | 2018–2019 | 2.4346% | -0.00205418 | 2 / 94（共 245） | 历史基准 |
| parameter-only `l24_strength050` | 2018–2019 | 1.5482% | -0.00001117 | 1 / 55 | FAIL |
| 全局 `l64_s08` | 2018–2019 | 3.3834% | +0.00154237 | 2 / 21 | 最新同口径结果 |
| 全局 `l64_s08` | 2020 prospective | 3.5734% | +0.00256941 | 0 / 10 | PASS |
| family/lead schedule | 2020 prospective | 3.6758% | +0.00297449 | 0 / 1 | PASS，次要分析 |

推荐 headline 仍是自由度较低的单一全局 `l64_s08`；family/lead schedule 保留为次要分析，不替代全局参数。2020 只新增 GraphCast 对照，不能与其他模型的 2018–2019 数值直接跨年份排名。

## 一条命令验证迁移结果

```bash
python -m pip install -r requirements-repro.txt
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py
```

验证器是只读的，并会完成：

- 校验 `MIGRATION_MANIFEST.sha256` 中每个文件；
- 确认迁移目录不含原始数据、缓存、压缩包或未列入允许清单的权重，并核对两份允许权重的固定大小与 SHA-256；
- 从冻结的 5,390 行 baseline CSV 与 490 行 SphereDyn CSV，逐字节重生成原版 49 表；
- 从最新版 5,880 行 CSV，逐字节重生成更新后的 49 表；
- 确认新旧结果之间只有 245 个 GraphCast + SphereTTC 行的 RMSE/ACC 改变；
- 从单元 CSV 重算原 GraphCast、parameter-only、`l64_s08` 回看、2020 prospective 与 family/lead schedule 汇总，并与冻结 JSON 对照。

成功输出以 `SOON_MIGRATION_REPRODUCIBLE` 开头。

## 权威结果入口

- 最新综合结论：`artifacts/graphcast_spherettc_20260804/LATEST_EXPERIMENT_SUMMARY_ZH.md`
- 最新 49 表：`artifacts/graphcast_spherettc_20260804/LATEST_RESULTS_49_VARIABLES_ZH.md`
- 最新机器可读 5,880 行：`artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`
- 原冻结 49 表：`artifacts/spheredyn_spherettc_open_goal_20260801/MAIN_RESULTS_49_VARIABLES_ZH.md`
- SphereDyn 主目标：`artifacts/spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/MAIN_GOAL_GATE.json`
- GraphCast 完整数值对比：`artifacts/graphcast_spherettc_20260804/COMPARISON_WITH_FROZEN_49_TABLES.json`
- 2020 prospective 报告：`artifacts/graphcast_spherettc_20260804/schedule/PROSPECTIVE_2020_REPORT.md`

## 目录内容

- `src/`：SOON、SphereDyn 与 SphereTTC 核心实现。
- `scripts/`：训练、推理、评估、表格生成和原完整归档验证脚本。
- `configs/`：统一实验配置。
- `tests/`：合成/单元测试；依赖原始 S2S 数据的两个 loader 测试在数据缺失时自动 skip。
- `artifacts/ttc_publication_corrected_20260726/`：原 11-baseline 冻结参数及聚合结果。
- `artifacts/spheredyn_spherettc_open_goal_20260801/`：SphereDyn 协议、主目标结论、聚合 CSV 与原版 49 表。
- `artifacts/graphcast_spherettc_20260804/`：parameter-only 与 schedule 代码/配置/审计、最新汇总、2020 prospective 结果和最新版 49 表。

两份允许携带的最终权重：

- SphereDyn-v9：`spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt`，16,302,127 字节；
- SphereTTC-v22：`main_prediction_v2_seed44/final/FITTED_WEIGHTS_2017.npz`，35,588,308 字节；
- 合计 51,890,435 字节，约 49.49 MiB。

## 复现边界

本迁移工程可以在没有原始数据的设备上确定性复现**所有已发布汇总数值和 49 张表**，并已携带最终 SphereDyn-v9 checkpoint 与 SphereTTC-v22 拟合权重。由于不携带原始数据，它不能单独从气象场重新计算预测或逐 initialization 指标；完整 GPU 重跑还需另行提供：

- ERA5/S2S 输入与 truth；
- GraphCast 和其他 baseline checkpoint；
- GraphCast 官方源码或兼容安装；
- ACC climatology 数组与预测缓存（若选择从缓存回放）。

接入这些外部资产后，可按 `scripts/`、`artifacts/graphcast_spherettc_20260804/*/tools/` 和各目录 `README.md` 中的入口重跑。历史日志/审计里的 `/mnt/...` 是原实验 provenance，不是离线汇总验证器的路径依赖。

## 可选代码测试

安装 `requirements.txt` 所列运行依赖后：

```bash
pytest -q
```

迁移包不写入权威结果；建议把新实验输出到单独工作目录。
