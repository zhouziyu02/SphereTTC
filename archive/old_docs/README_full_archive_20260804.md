# SOON 最新完整实验复现仓库

本目录已汇总原 49 变量主实验，以及 2026-08-04 完成的两组 GraphCast + SphereTTC 后续实验。仓库内代码、冻结逐 initialization 指标、配置、报告、GraphCast checkpoint、2019 warmup、完整 2020 预测缓存和 2020–2021-01-10 truth 均使用仓库相对路径组织，可整体迁移。

## 最新结论

- 原 SphereDyn-v9 + SphereTTC-v22 主目标保持完成：2018–2019 macro nRMSE 从 0.3808743 降至 0.2845782，改善 25.2829%。
- 参数保守实验 `l24_strength050` 未达到预注册的 2% 门槛：RMSE 改善 1.5482%，macro ACC -0.00001117，判定 FAIL。
- 后续推荐结果是单一全局 `l64_s08`。在与原 49 表相同的 2018–2019 口径上，RMSE 改善由原 GraphCast + SphereTTC 的 2.4346% 提高到 3.3834%，macro ACC 由 -0.00205418 转为 +0.00154237。
- `l64_s08` 在独立 2020 prospective holdout 上 PASS：RMSE 改善 3.5734%，0/245 个 RMSE 单元退化，macro ACC +0.00256941。
- family/lead schedule 的 2020 结果为 3.6758%，只作为自由度更高的次要分析。

## 结果入口

- 最新总览：`artifacts/graphcast_spherettc_20260804/LATEST_EXPERIMENT_SUMMARY_ZH.md`
- 更新 GraphCast 行后的最新 49 表：`artifacts/graphcast_spherettc_20260804/LATEST_RESULTS_49_VARIABLES_ZH.md`
- 最新机器可读 5,880 行：`artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`
- 原冻结 49 表：`artifacts/spheredyn_spherettc_open_goal_20260801/MAIN_RESULTS_49_VARIABLES_ZH.md`
- 2020 prospective 报告：`artifacts/graphcast_spherettc_20260804/schedule/PROSPECTIVE_2020_REPORT.md`
- 完整数值对比：`artifacts/graphcast_spherettc_20260804/COMPARISON_WITH_FROZEN_49_TABLES.json`

原 49 表继续作为不可变历史基准；最新 49 表只替换 GraphCast + SphereTTC 的 245 个 2018–2019 单元，其他 5,635 行保持不变。2020 结果不与其他模型的 2018–2019 数值跨年份混表。

## 迁移后验证

在仓库根目录执行：

```bash
python -m pip install -r requirements-repro.txt
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_latest_results_reproducibility.py
```

成功时同时出现：

```text
MAIN_RESULTS_REPRODUCIBLE ...
LATEST_RESULTS_REPRODUCIBLE rows=5880 variables=49 tables=49 ...
```

该命令只读地完成以下工作，不训练、不推理、不覆盖冻结结果：

1. 从冻结逐 initialization 指标逐字节重建原 49 表；
2. 重算 parameter-only 的 2017 选择和 2018–2019 FAIL 结论；
3. 从 56 个候选重做 `l64_s08` 的 2017 双窗口选择；
4. 重算 family/lead schedule、2018–2019 回看、2020 prospective 和固定种子的 2,000 次 block bootstrap；
5. 逐字节重建最新 5,880 行 CSV 和 49 张 Markdown 表；
6. 检查必要的 GraphCast checkpoint、预测缓存和 truth 资产。

若还要流式校验归并实验目录内全部约 14 GB 文件：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_latest_results_reproducibility.py --full-integrity
```

GPU 端到端重跑说明、已验证环境和历史命令见 `artifacts/graphcast_spherettc_20260804/REPRODUCIBILITY.md`。历史日志和 profile 中的绝对路径仅记录原执行环境，不再作为运行依赖。
