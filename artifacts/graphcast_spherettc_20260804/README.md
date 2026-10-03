# GraphCast + SphereTTC 后续实验归档

本目录完整归并两个原 sibling 实验：

- `parameter_only/`：保守参数搜索，最终未达到预注册门槛，保留为负结果和参数选择证据。
- `schedule/`：56 个全局候选、family/lead schedule、2018–2019 冻结后回看与独立 2020 prospective 实验；推荐全局 `l64_s08`。

归并时保留了所有配置、逐 initialization 指标、逐单元 CSV、日志、profile、完整 2019 warmup、2020 GraphCast 缓存和专用 truth。工具已从 sibling 固定路径改为向上自动发现 `SOON` 根目录；历史日志、profile 和旧 `AUDIT.json` 中的绝对路径只作为 provenance 保留。

权威入口：

- `LATEST_EXPERIMENT_SUMMARY_ZH.md`
- `LATEST_RESULTS_49_VARIABLES_ZH.md`
- `LATEST_RESULTS.csv`
- `COMPARISON_WITH_FROZEN_49_TABLES.json`
- `schedule/MAIN_EXPERIMENT_RECOMMENDATION.md`
- `LATEST_REPRODUCIBILITY_MANIFEST.json`

从仓库根目录运行 `python scripts/verify_latest_results_reproducibility.py` 做只读统计复现；追加 `--full-integrity` 会校验整个归并目录的树哈希。
