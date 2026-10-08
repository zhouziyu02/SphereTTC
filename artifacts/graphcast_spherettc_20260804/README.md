# GraphCast + SphereTTC 冻结实验

这里保留 GraphCast 的 SphereTTC 参数更新和评估证据。`schedule/` 的 56 个候选仅使用 2017 搜索段与时间后置 holdout，冻结全局 `l64_s08` 后评估 2018–2019 和独立的 2020 年。`parameter_only/` 保留早期负结果。

最新 `LATEST_RESULTS.csv` 只含 11-backbone SphereTTC 的 5,390 行。相对 publication 结果仅替换 GraphCast + SphereTTC 的 245 个单元，所有 Raw 和其他 backbone 指标原样保留。`LATEST_RESULTS_49_VARIABLES_ZH.md` 包含 49 张表，每张 22 行。

保留的配置、报告、逐单元 CSV、日志和 profiles 是冻结证据。预测缓存、逐 initialization 指标、truth 和官方 checkpoint 不包含在轻量仓库中；历史 GPU 工具需要外部资产才能运行。完整性快照已移入 `../provenance/graphcast_spherettc_20260804/`，原服务器绝对路径仅供历史追溯。当前验证与重建入口见 `REPRODUCIBILITY.md`。
