# 主实验工程清理记录

清理日期：2026-08-03（UTC）

## 基准

唯一权威结果为 `MAIN_RESULTS_49_VARIABLES_ZH.md`，SHA-256：

`2aad727b4ca3bd8cab9da041632d867cc9764beb4bede694c9a5b5f8dbb1aca1`

## 工程内保留内容

- `artifacts/` 仅保留三个主链路根目录：本地主模型 baseline 缓存、官方 baseline/SphereTTC 指标，以及 SphereDyn/SphereTTC-v22 主结果；
- 11-baseline 的 publication SphereTTC 冻结逐 initialization 指标、结果、参数、2017–2019 预测缓存及 ACC climatology；
- 六个本地 baseline 的 checkpoint 与 2017–2019 预测缓存；
- SphereDyn-v9 seed 44 唯一主 checkpoint、2017–2019 Raw 缓存、SphereTTC-v22 Final 缓存、逐样本指标、门槛与 manifest；
- 1979-01-01 至 2020-01-10 的 S2S 数据，用于覆盖 fit、2017 参数选择、2018–2019 测试和最长 240h target；
- 主实验运行、统计、表格生成和复现校验所需的最小脚本、配置、源码和测试；
- 11 个 baseline 所需的外部实现与权重。

## 工程内移除内容

- 历史搜索、早期 SphereTTC v2–v21、消融、诊断、多 seed、2020/2023 prospective 与失败候选产物；
- 非主 SphereDyn checkpoint、预测缓存、指标、队列、日志和 profile；
- 被最终逐样本指标取代的临时运行目录、旧报告、旧训练入口和缓存；
- 2020-01-10 之后的数据、2021–2023 及单独 prospective 2023 数据；
- GenCast checkpoint、重复的外部模型资产、示例 notebook/图片及无关的大型测试数据；
- 旧多模型诊断 CLI 和不再被主链路调用的历史源码分支；
- Python bytecode、pytest cache、`.lock` 文件、空 Git 占位目录及外部仓库版本控制元数据。

高风险内容没有直接永久擦除，而是原子移动到工程外的可恢复归档：

`/mnt/bn/czl-no-conda/mlx/users/ziyuzhou/SOON_CLEANUP_ARCHIVE_20260803`

该归档不属于清理后的 SOON 工程；如需永久释放其占用空间，应在明确确认不可恢复后单独删除。

## 清理后验证

运行 `PYTHONDONTWRITEBYTECODE=1 python scripts/verify_main_results_reproducibility.py` 可核对全部冻结 SHA-256，并从逐 initialization 指标逐字节重建 5,390 行 baseline CSV、490 行 SphereDyn CSV 和最终 49 表 Markdown。清理过程中没有训练或预测。
