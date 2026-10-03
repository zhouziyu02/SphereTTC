# 05 · 论文叙事：参照 PnP-Corrector（ICML 2026）

**定位**：本工作是 test-time calibration / test-time computing 在 medium-range 天气（气候）预报上的第一次系统尝试。两个 SOTA 分工不同、并不冲突：

- **SOTA A**：SphereTTC 作为即插即用的 calibrator，能稳定提升现有 baseline。它回答的问题是：TTC 对这个任务是否有效、是否通用。
- **SOTA B**：SphereDyn + SphereTTC。它说明 SphereTTC 背后的球面机理对这个任务有用，同时我们提出的 SphereDyn 本身就是一个强 baseline。它回答的问题是：这套思想能否支撑一个完整的系统。

本文件把 PnP-Corrector 的论证骨架对应到本仓库的证据上，并标出现在已经有什么、还缺什么。

## 1. PnP-Corrector 的论证骨架

PnP-Corrector: A Universal Correction Framework for Coupled Spatiotemporal Forecasting（ICML 2026，arXiv 2605.08935）的三条贡献是：

1. **New Problem Formulation**：首次提出并形式化耦合时空预报中的 Reciprocal Error Amplification；
2. **Superior Performance**：设计自有 backbone DSLCast（含 differentiable semi-Lagrangian advection），在耦合地球系统预报上达到 SOTA；
3. **Novel Framework**：PnP-Corrector 是模型无关、即插即用的 correction agent。它冻结预训练模拟器，只训练校正器。

它的实验结构如下：

- 主表：6 个 backbone（ConvLSTM、SimVP、GraphCast、Ola、CirT、DSLCast），列出多个 lead 下的 RMSE/MAE、加校正器后的值和相对提升；
- ACC 表；
- 自有 backbone 在加校正器后取得最佳绝对性能，但相对提升较小（1–2%）；
- 消融：backbone 组件逐个去掉，以及校正器开/关；
- 效率表：参数量、MACs、训练时间；
- 长时效曲线、能量谱、极端事件 CSI/SEDI、可视化 case study；
- 扩展实验：再加入陆面耦合。

值得注意的是：它的 baseline 都按同一套协议重新训练（包括 GraphCast）；校正器需要训练（文中约 64 张 A100 训练 22 小时）；它没有和其他校正方法做实证对比。

## 2. 逐项对应

| PnP-Corrector | SphereCast（本工作） | 本仓库证据 |
|---|---|---|
| 新问题：耦合系统的误差相互放大 | 新问题：中期预报中的 test-time calibration，在 **delayed verification** 下只用已核验的“预报–观测”对在线校准冻结的预报系统 | 协议：`configs/`、`src/ttc/memory.py`；注意 R1 |
| 即插即用校正器（需要训练） | **SphereTTC**：即插即用、**无需梯度训练**、因果的 test-time calibrator | `src/spherettc.py`；`results/EFFICIENCY.md` |
| 自有 backbone DSLCast（semi-Lagrangian advection） | 自有 backbone **SphereDyn**（semi-Lagrangian 输运、截断球谐谱动力学、球面局部块、谐波气候趋势、多尺度球面残差） | `src/spheredyn.py` |
| 主表：6 个 backbone × 多个 lead，Raw / +校正 / 相对提升 | **SOTA A**：11 个 backbone × 5 个 lead | `results/PAPER_TABLE1_NRMSE_BY_LEAD.md`、`PAPER_TABLE2_ACC_BY_LEAD.md` |
| DSLCast + 校正器取得最佳绝对性能 | **SOTA B**：SphereDyn + SphereTTC 的 macro ACC 全表第一，168h/240h 的 nRMSE 与 ACC 均为全表第一 | `results/SPHEREDYN_SPHERETTC.md` |
| DSLCast 优于同协议 baseline | Raw SphereDyn 在**每个 lead、nRMSE 与 ACC 两个指标上**都优于全部 6 个同协议训练的 baseline | 同上 |
| 效率表 | SphereTTC 无可学习参数；每次预报额外开销约 0.07 s，约为 GraphCast 推理时间的 0.22% | `results/EFFICIENCY.md` |
| 扩展实验（加入陆面） | **2020 prospective 部署模拟**：参数冻结后，用官方 checkpoint 重新生成全年预报再校准，0/245 个单元变差 | `artifacts/graphcast_spherettc_20260804/schedule/PROSPECTIVE_2020_REPORT.md` |

相对 PnP-Corrector，我们可以强调的优势：

- **不需要训练**：只在 2017 年做超参选择，每个候选在 V100 上约 1 分钟；
- **严格因果**：只用已核验的样本；
- **直接作用于官方发布的冻结模型**：GraphCast、Pangu、FuXi、OneForecast、FourCastNetV2 都不重新训练；
- **有独立年份的 prospective 验证。**

相对 ST-TTC（NeurIPS 2025，“Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting”），我们的差异在于：

- ST-TTC 的实验是交通、空气质量和能源数据，**没有全球中期天气预报**；
- 我们面对的是球面几何、多日 lead 的延迟监督、49 个变量，以及业务级 AI 天气模型。

## 3. 两个 SOTA 在论文中的分工

### SOTA A：SphereTTC 作为 calibrator 能有效提升现有 baseline

- 11/11 个 backbone 的 macro nRMSE 下降，平均 4.20%，中位数 1.34%；5 个 lead 上的平均变化全部为负（24h −6.28%，72h −3.95%，120h −2.99%，168h −3.34%，240h −4.54%）。
- 覆盖面：同协议训练的模型（ConvLSTM、Transformer、FNO、ViT、CirT、ClimODE）和官方发布模型（FourCastNetV2、OneForecast、FuXi、Pangu、GraphCast）都有提升。即使是最强的 GraphCast，也下降了 4.57%。
- ACC：9/11 个 backbone 上升，OneForecast 和 Pangu 略降，需要如实写出。
- 稳健性：GraphCast 的 2020 prospective 结果为 0/245 个单元变差，95% CI 为 [3.42%, 3.73%]。

### SOTA B：球面机理有用，SphereDyn 是 SOTA baseline

建议的论证链（与 PnP 用 DSLCast 的方式一致）：

1. **SphereDyn 是同协议下的 SOTA baseline**：Raw SphereDyn（0.3809）在 5 个 lead 上的 nRMSE 和 ACC 都优于全部 6 个同协议 baseline（最好的 FNO 为 0.4427）。这和 PnP 中 DSLCast 对比重新训练过的 baseline 是同一种比较方式。
2. **官方发布模型单独列为一组**：GraphCast、Pangu 等在 0.25°、6 小时 ERA5 上以大得多的算力训练，在本文中冻结使用。Raw SphereDyn 的平均水平不如它们（GraphCast 为 0.2917），所以“SOTA baseline”这个说法要限定在“同协议训练”的范围内。Raw SphereDyn 在 240h 的 nRMSE 是全部 12 个 Raw 模型里最低的，但它在 240h 的 ACC 只排第 6/12，说明这一项优势部分来自预报更平滑，不宜作为主要论点。
3. **SphereDyn + SphereTTC 是最完整的系统**：macro nRMSE 0.2846，低于所有 Raw 模型；macro ACC 0.8080，全表最高；168h 和 240h 在两个指标上都是全表第一；245/245 个单元的 RMSE 与 ACC 都提升。
4. **“球面机理有用”的直接证据还需要补**：
   - SphereTTC 有/无球谐表示的对比：`run_ttc.py --method affine_ttc` 就是格点空间的仿射校准，正好作为对照；
   - SphereDyn 的组件消融：v9 的设计里已经有 null control（`DESIGN.json`），需要重新训练；
   - SphereDyn + 单 backbone SphereTTC 的结果：使 SOTA B 和 SOTA A 用同一种 SphereTTC 算子，并把 25.28% 中“来自参考预报组合”的部分拆出来（见 `04_AUDIT_AND_RISKS.md` R2）。

## 4. 贡献表述草稿（英文，可直接改写）

> 以下数字是修复 R1 之前的结果，投稿前需要更新。

1. **Problem.** We formulate *test-time calibration for medium-range weather forecasting*: a frozen forecasting system is corrected online at every initialization using only forecast–observation pairs whose verification is already available, under the delayed supervision inherent to forecasting. To our knowledge, this is the first systematic study of test-time calibration (test-time computing) for global medium-range forecasting with modern AI weather models.
2. **Method.** We propose **SphereTTC**, a training-free, model-agnostic calibrator. It estimates conservative affine corrections in a truncated spherical-harmonic space with recency- and robustness-weighted delayed memory, and guards them with a chronological validation gate and a no-op fallback. When multiple forecast providers are available, SphereTTC extends to lead- and variable-specific constrained combinations that preserve the target backbone.
3. **Backbone.** Following the same spherical principle, we design **SphereDyn**, a spherical dynamics backbone combining semi-Lagrangian transport, truncated spherical-spectral dynamics and a multi-resolution spherical residual pathway. SphereDyn outperforms all baselines trained under the same protocol at every lead time in both RMSE and ACC.
4. **Results.** On ERA5 (2018–2019; 49 variables; 1–10-day leads), SphereTTC reduces the macro normalized RMSE of all 11 backbones, including GraphCast, Pangu and FuXi used as frozen released systems, by 4.2% on average, with an average improvement at every lead. It adds about 0.07 s per forecast (≈0.2% of GraphCast inference). SphereDyn + SphereTTC achieves the highest anomaly correlation among all 24 configurations, and the best RMSE and ACC at 7- and 10-day leads. A prospective evaluation of GraphCast + SphereTTC on 2020 with frozen parameters shows no RMSE regression in any of the 245 variable–lead cells.

## 5. 实验章节规划（对照 PnP-Corrector）

| 编号 | 内容 | 状态 | 文件 / 需要做什么 |
|---|---|---|---|
| Tab. 1 | macro nRMSE：12 个 backbone × 5 个 lead，Raw → +SphereTTC（相对变化）。建议分两组：同协议训练（含 SphereDyn）与官方发布模型 | ✅ 草稿已生成 | `results/PAPER_TABLE1_NRMSE_BY_LEAD.md` |
| Tab. 2 | macro ACC：同上 | ✅ | `results/PAPER_TABLE2_ACC_BY_LEAD.md` |
| Tab. 3 | Z500 / T850 / T2M / U10 在 120h 与 240h | ✅ | `results/HEADLINE_120H_LATEST.md`、`HEADLINE_240H_LATEST.md` |
| Tab. 4 | **与其他 test-time / 偏差校正方法对比**：static bias、decaying-average（EMA）bias、格点 affine、geo、zonal spectral、ST-TTC | ❌ 代码已有（`run_ttc.py` 的各个 `--method`），需要在已有缓存上运行，成本低 | 这是 PnP 没做、但审稿人最可能要求的对比 |
| Tab. 5 | SphereTTC 消融：去掉球谐（`affine_ttc`）、去掉 gate、去掉 robust 或 recency 加权、不同 lmax / strength | ⚠️ GraphCast 上已有 2017 年 56 个候选的敏感性数据；组件消融还要跑 | `artifacts/graphcast_spherettc_20260804/schedule/VALIDATION_CANDIDATES_2017.csv` |
| Tab. 6 | SphereDyn 组件消融（null control、去掉 semi-Lagrangian 或谱动力学） | ❌ 需要训练 | `DESIGN.json` 里已定义 null control |
| Tab. 7 | SOTA B 拆分：SphereDyn + 单 backbone SphereTTC；只用参考预报的组合；SphereDyn + 组合 | ❌ 成本低 | R2 |
| Tab. 8 | 效率：参数量、校准开销、显存、超参选择成本 | ✅ 部分；缺 MACs 与 SphereDyn 的总训练 GPU 小时 | `results/EFFICIENCY.md` |
| Fig. | 每个变量按 lead 的误差曲线（Raw vs +SphereTTC） | ✅ 可以直接从 CSV 画 | `artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv` |
| Fig. | 2020 prospective 与 bootstrap 区间 | ✅ | `schedule/PROSPECTIVE_2020_*` |
| Fig. | 球谐功率谱：校准前后误差的谱分布，说明 SphereTTC 主要修正低阶、相干的误差 | ❌ 需要原服务器上的预测缓存 | 最能直观说明“球面机理”的一张图 |
| Fig. | 极端事件 CSI / SEDI、典型个例地图、gate 随时间的变化 | ❌ 需要预测缓存 | 对应 PnP 的极端事件分析与可视化 |

## 6. 措辞边界

- **“first”**：限定为“首次系统研究 AI 天气模型的全球中期 test-time calibration”。业务数值预报里早就有在线偏差订正，例如 NCEP NAEFS 的 decaying-average bias correction、Kalman 滤波后处理和 MOS。Related work 必须讨论这些方法，Tab. 4 里的 `ema_bias` 实际上就是 decaying-average 方法，可以直接作为对比。投稿前还需要做一次系统的文献检索。
- **“SOTA”**：SphereDyn 写成“在同一训练协议下优于所有 baseline”；SphereDyn + SphereTTC 写成“在全部 24 种配置中 ACC 最高、7–10 天 RMSE 与 ACC 最佳”。
- **SOTA B 用的是 SphereTTC 的多提供方模式**：表格脚注要写明，正文里同时给出单 backbone 模式的结果（Tab. 7）。
- **数字的有效性**：R1 修复并重跑之前，A 中 5 个官方模型的数字都是暂定的。
