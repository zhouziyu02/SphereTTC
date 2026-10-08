# 05 · SphereTTC 论文叙事与后续实验

论文聚焦一个问题：冻结的全球中期预报模型，能否利用随后才可获得的真值，在不进行梯度训练的前提下保守地校正后续预报？方法和实验都围绕单 backbone SphereTTC 展开。

当前主证据为 11 个异构 backbone、五个时效、49 个变量的 Raw / calibrated 对照。**现有官方模型数字有 R1 日均真值可用时刻问题，投稿前必须修复并重跑。** 以下是写作结构与待验证贡献，不把工作计划当成已完成结果。

## 1. 论证结构

1. **任务与协议。** 明确 initialization、valid time、truth availability、观测延迟和 warmup；把 delayed verification 作为约束，而非仅靠日期标签声称因果性。
2. **方法。** 截断球谐表示、向恒等收缩的仿射拟合、recency / robust 加权、chronological validation gate、degree taper 和 no-op。每个 lead 与变量独立校准，backbone 保持冻结。
3. **跨模型证据。** 同协议训练的六个本地 backbone 与五个冻结官方系统分组展示，避免暗示训练预算相同。报告 macro nRMSE、ACC、每时效结果和逐单元回退。
4. **为何有效。** 用 grid affine 等对照、受控组件消融和误差谱分析检验球面表示与风险控制的贡献；当前代码与主表本身不能证明各组件必要。
5. **可靠性与成本。** 用冻结未来年份、观测延迟敏感性、时间块 bootstrap 和 profile 说明适用范围，同时披露失败候选、选参历史和计时边界。

## 2. 当前可陈述的事实

在原始协议的冻结记录中，11/11 个 backbone 的 macro nRMSE 下降，平均 4.20%、中位 1.34%；五个时效的平均增益为正。macro ACC 在 9/11 个 backbone 上升，OneForecast 与 Pangu 略降，且部分变量–时效单元出现负迁移。

GraphCast `l64_s08` 的 2018–2019 macro nRMSE 变化为 −4.57%。后续 2020 记录中没有 RMSE 回退单元（0/245），平均单元增益 3.57%（14 天 block-bootstrap 95% CI [3.42%, 3.73%]）。这些数字均需在 R1 修复后重新确认；2020 结果应单独报告，不能与其他模型 2018–2019 排名混用。

A100 profile 显示约 0.07 s / initialization 校准增量开销；含 truth I/O 与 scoring 约 0.36 s。约 0.2% 的推理成本占比仅相对于记录中的 GraphCast-small。当前没有其他十个 backbone 的完整计时或 MACs/FLOPs。

## 3. 英文贡献草稿

以下句子刻意区分实现、研究目标和待复核结果；最终论文数字须更新为修复协议后的结果。

> We study test-time calibration of frozen global medium-range weather forecasts under delayed verification. We make the availability of forecast–observation pairs an explicit part of the evaluation protocol.
>
> We introduce SphereTTC, a model-agnostic calibrator that fits conservative affine corrections in a truncated spherical-harmonic space. Recency and robustness weighting, chronological validation and a no-op fallback control the correction without gradient training of the forecasting backbone.
>
> We evaluate the approach across six locally trained backbones and five released forecasting systems using common variables, leads and daily-mean metrics. The retained experiments provide initial evidence of macro error reduction, with exceptions in ACC and individual variable–lead cells; final causal conclusions require correction of the documented daily-mean truth-availability issue and rerunning the affected evaluations.

不在主方法贡献中写多时间尺度 / 季节专家；它们只有实验代码，没有冻结主结果。新颖性和相对于其他方法的优越性，应在文献核对、对照实验之后另行形成结论。

## 4. 实验与图表计划

| 内容 | 当前状态 | 下一步 / 证据 |
|---|---|---|
| 主表：11 backbone × 五个 lead，Raw / SphereTTC nRMSE | 已有冻结草稿 | `results/PAPER_TABLE1_NRMSE_BY_LEAD.md`；修复 R1 后更新 |
| ACC 表、逐单元改善 / 回退数 | 已有冻结草稿 | `results/PAPER_TABLE2_ACC_BY_LEAD.md`、`SPHERETTC_11_BACKBONES.md` |
| Z500 / T850 / T2M / U10，120 / 240 h | 已有冻结草稿 | `results/HEADLINE_120H_LATEST.md`、`HEADLINE_240H_LATEST.md` |
| 同协议 TTC / 偏差校正比较 | 已有代码，无冻结结果 | `static_bias`、`ema_bias`、`affine_ttc`、geo、zonal spectral、ST-TTC |
| SphereTTC 组件消融 | 尚缺 | 去除或替换球谐、gate、recency、robust、taper；每项使用相同 truth availability 和选参预算 |
| 超参敏感性 / 统一选参 | GraphCast 有 2017 的 56 候选 | `schedule/VALIDATION_CANDIDATES_2017.csv`；扩展 lmax 上界，统一 11 backbone 协议 |
| 真值可用延迟敏感性 | 尚缺 | 日均结束后准入及额外业务延迟；报告不同 memory 可用量 |
| q 转换审计与剔除 q 后指标 | 尚缺 | 核对 RH 单位、饱和水汽压公式和压力单位 |
| 独立年份 / prospective 与时间块区间 | GraphCast 2020 已有原协议记录 | 修复后复核；未来年份不可再参与开发 |
| 效率 | 有 GraphCast profile | `results/EFFICIENCY.md`；补其他模型计时和内存 / I/O 边界 |
| 误差随 lead 变化 | 可由 CSV 绘制 | `artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv` |
| 校准前后误差功率谱、典型个例地图 | 需预测缓存 | 检验低阶相干误差的变化，而非仅展示好看的案例 |
| 极端事件、gate 时间序列、区域 / 季节分析 | 需预测缓存与新评估 | 同时报告失败模式，避免以平均误差代替全部业务效用 |

## 5. 写作边界

- 不把 11/11 macro 改善写成所有变量与时效改善；不要略去两项 ACC 回退。
- 不把 GraphCast 2018–2019 当作全程盲测：扩大候选空间前已查看测试结果。
- 不将低开销解释为无拟合：仿射系数在线估计，但没有梯度训练 backbone。
- 不直接比较日均 1.5° GraphCast-small 与 WeatherBench2 的瞬时高分辨率 headline。
- 对既有时空 TTC 与 NWP 偏差订正做文献核对；当前仓库本身不能证明“首次”或 calibration SOTA。
- 先解决 R1，再补对照和消融；单纯重排表格或清理目录不会提高证据可信度。
