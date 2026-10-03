# SphereDyn + SphereTTC 主实验总结

## 1. 冻结目标

主实验只使用一个随机种子 seed 44，不要求多 seed、消融或额外分析。以 2018–2019 的主预测结果为最终评价，要求 SphereDyn + SphereTTC 的 macro nRMSE 优于同次 Raw SphereDyn，最低提升门槛为 5%。GraphCast 只作为报告参考，不是硬门槛。

## 2. 统一评估设置

| 项目 | 设置 |
|---|---|
| 测试时期 | 2018-01-01 至 2019-12-31 |
| Initialization | 730 个逐日初始化 |
| 变量 | common-49：45 个高空变量与 4 个地面变量 |
| 时效 | 24、72、120、168、240h |
| 指标 | 余弦纬度加权物理量 RMSE、ACC、macro nRMSE |
| ACC climatology | 仅由 1979–2016 fit split 构建的逐月日气候态 |
| SphereTTC 参数选择 | 仅使用 2017 |
| 测试策略 | 参数冻结后评价 2018–2019 |

主表包含原 11 个 baseline：ConvLSTM、Transformer、FNO、ViT、CirT、ClimODE-style adapter、FourCastNetV2、OneForecast、FuXi、Pangu 和 GraphCast，以及 SphereDyn-v9 seed 44。原 11 个 baseline 各报告 Raw 与 publication SphereTTC；SphereDyn 报告 Raw 与 SphereTTC-v22。

## 3. SphereDyn 主预测

| 项目 | 值 |
|---|---:|
| checkpoint | `spheredyn_v9_multiscale.pt` |
| checkpoint SHA-256 | `121f9b900db554f175f121e629ac64a10ce2697e7cf4bfd224f09984c6bbf453` |
| Raw SphereDyn macro nRMSE | 0.3808743034930633 |
| SphereDyn + SphereTTC-v22 macro nRMSE | 0.28457822236659047 |
| 相对提升 | 25.282903110901678% |
| 5% 主门槛 | PASS |
| GraphCast reporting-only macro nRMSE | 0.2917217030977239 |

SphereTTC-v22 在全部 245 个“变量 × 时效”单元中均降低 RMSE，并在全部 245 个单元中提高 ACC。

## 4. 49 变量主表

权威表格为 `MAIN_RESULTS_49_VARIABLES_ZH.md`。它由以下冻结来源生成：

1. `../ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv`：11 baseline × 2 方法 × 49 变量 × 5 时效，共 5,390 行。
2. `main_prediction_v2_seed44/paper_metrics/SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv`：SphereDyn × 2 方法 × 49 变量 × 5 时效，共 490 行。

两者合计 5,880 条结果。每个变量一张表，共 49 张；每张表 24 行模型/方法和 10 个数值列。

## 5. 可复现性

冻结的 11-baseline 与 SphereDyn 逐 initialization 指标均已保留。运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_main_results_reproducibility.py
```

验证程序会执行以下只读/临时操作：

1. 核对 checkpoint、climatology、逐样本指标、CSV 和最终 Markdown 的 SHA-256；
2. 从 22 份 baseline 指标重新聚合 5,390 行 CSV；
3. 从 SphereDyn Raw/Final 指标重新聚合 490 行 CSV；
4. 合并两份结果并重建 49 张表；
5. 要求两份 CSV 和最终 Markdown 与冻结文件逐字节一致。

该流程只做确定性统计聚合，不训练、不运行预测，也不修改冻结结果。

## 6. 解释边界

- 只有 seed 44，不能声称跨随机种子稳定性。
- 没有消融，不能独立量化每个 SphereTTC 组成部分的因果贡献。
- SphereTTC-v22 使用五个外部参考预测，不能把最终结果归因于纯 SphereDyn。
- GraphCast 是组合输入之一，因此不能据此声称独立击败 GraphCast。
- 当前主实验目标已经完成；不再安排新实验。
