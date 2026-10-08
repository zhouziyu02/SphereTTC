# SphereTTC 表格与冻结证据复现

从仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py
python scripts/tables/build_latest_results.py --check
python tools/build_results_summary.py --check
```

本轻量仓库可自包含地核对 5,390 行最新结果、11 个 backbone、22 个模型/方法配置、49 张表和 GraphCast 的冻结标量摘要。构建器从原 publication CSV 与冻结 `GRAPHCAST_L64_S08_2018_2019.csv` 组合结果，保留每个指标字符串。共享标准差为 `../shared/s2s_daily_54var_stats.json`。

`parameter_only/` 和 `schedule/` 保留 GraphCast 选参、评估、bootstrap 的报告、配置、逐单元 CSV 与运行 profiles；这些是冻结证据。原始数据、预测缓存、逐 initialization NPZ 指标和官方 checkpoint 未打包，因此不能在本仓库中重新聚合逐样本指标、重跑参数搜索或 bootstrap，也不能直接重跑 GPU 推理。历史研究工具保留供参考，使用前需重新准备相应外部数据与运行环境，并将输出指向独立实验目录。

历史完整性快照和审计位于 `../provenance/graphcast_spherettc_20260804/`，只描述原环境。它们的旧绝对路径或被移除文件名不是当前运行依赖。日常验证以根目录 `verify_migration.py` 为准。
