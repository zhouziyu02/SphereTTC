# 04 · 投稿前审计：风险与必须补的实验

这份审计按“顶会审稿人会不会质疑”的标准来写，针对 `01_SOTA_RESULT.md` 中的两个 SOTA：**A. 最终版 SphereTTC 在 11 个 backbone 上的一致提升**，**B. SphereDyn + SphereTTC**。结论先放前面：两个结果都可以作为主实验，但 **R1 必须先修复并重跑**（影响 A 中 5 个官方模型）；B 需要补两个拆分对照（R2）和一次干净的冻结评估（R3），proposal 里的个别组件也需要与实际实现对齐（R11）。

| 编号 | 严重度 | 问题 | 影响的结果 |
|---|---|---|---|
| R1 | 高 | 日均目标进入 memory 时最多有 18h 前视 | A 中 5 个官方模型（含 GraphCast `l64_s08` 与 2020 prospective） |
| R2 | 高 | B 的增益里包含与 5 个参考预报的组合，且 v22 权重 2017 拟合后不再在线更新，需要拆分增益来源 | B：SphereDyn + SphereTTC 的 25.28% |
| R3 | 高 | SphereDyn 线用 2018–2019 做过开发和选择，协议中途改过 | B 全部结果 |
| R4 | 中 | GraphCast 线的 2018–2019 测试集被反复查看 | A 中 GraphCast 一行 |
| R5 | 中 | FourCastNetV2 / FuXi 的大增益集中在由 RH 换算出的 q | FCNv2 19.8%、FuXi 9.8% |
| R6 | 中 | publication 设置在强模型上降低 macro ACC；11 模型的超参协议不统一 | 主表 SphereTTC 行 |
| R7 | 中 | 没有 TTC baseline 对照、没有消融，只有单 seed | 整篇论文的实验部分 |
| R8 | 低 | 本包缺少原始数据、缓存和训练代码，部分依赖未记录 | 端到端复现 |
| R9 | 低 | 选中的 `lmax=64` 在搜索网格边界 | 超参敏感性 |
| R10 | 低 | 评估口径（日均、1.5°、GraphCast-small）与 WB2 headline 不同 | 与已发表数字的可比性 |
| R11 | 中 | proposal 中的 multi-timescale / seasonal experts 没有进入任何最终结果；B 所用的 v22 不含球谐分量 | 方法描述与实验的一致性 |
| R12 | 中 | “first” 和 “SOTA” 的论断范围：业务 NWP 早有在线偏差订正；SphereDyn 只在同协议下是 SOTA | 摘要与贡献表述 |

---

## R1（高）日均目标的 delayed memory 最多前视 18 小时

**现象。** 官方模型（FourCastNetV2、OneForecast、FuXi、Pangu、GraphCast）在每天 00 UTC 初始化；lead *L* 的预报值是 *init+L* 当天 00/06/12/18 UTC 四个时刻的平均，truth 是同一 UTC 日的日均值（`configs/official_daily_common49_2018_2019.yaml`：`initialization_hour_utc: 0`、`daily_mean_sample_hours_utc: [0, 6, 12, 18]`）。但 memory 的准入条件是 `valid_time <= current_init`：

- `src/spherettc.py::eligible_from_buffer`：`np.datetime64(item[0]) <= current`（`scripts/run_ttc.py` 的所有在线方法都从这里导入）
- `src/spherettc.py::EligibleMemory`：`valid_times[:, lead] <= current`
- `tests/test_no_leakage_memory.py::test_valid_time_equal_current_init_is_eligible` 还把这条规则写成了测试

于是在 *D* 日 00 UTC 校准时，每个 lead 都会用到一对 truth 为 *D* 日日均值的样本，而这个日均值包含 *D* 日 06/12/18 UTC 的观测，相对当前初始化时刻是未来 6–18 小时的信息。这一对正好是 recency 权重最大的一条，也落在 chronological gate 的 holdout 段里。

**不受影响的部分。** SphereTTC-v22（静态拟合，无 memory）不受影响。本地 6 个 baseline 和 SphereDyn 的输入是 S2S 日均场，若把 *D* 日的发布时刻定义为 *D* 日结束，则不存在这个问题；这一点需要在论文里写清楚。

**修复。** 对日均目标，准入条件改为 `valid_time + 1 day <= current_init`（逐日 cadence 下等价于严格 `<`）。改 `src/spherettc.py` 里的 `eligible_from_buffer` 和 `EligibleMemory`（2026-10-03 合并后这是唯一位置），并更新 `tests/test_no_leakage_memory.py`。然后用原服务器上已有的预报缓存重跑：

1. GraphCast `l64_s08` 的 2017 选择（56 个候选，`artifacts/graphcast_spherettc_20260804/schedule/tools/search_schedule.py`）；
2. 2018–2019 回看（`run_retrospective_primary.py`）和 2020 prospective（`run_prospective_holdout.py`）；
3. 其余 10 个 backbone 的 SphereTTC（至少 4 个官方模型）。

单个 backbone 一年的 SphereTTC 在 A100 上约 2 分钟（`profiles/retrospective_l64_s08_2018.json`：125.7 s），整套重跑的成本很低。预计影响不大，但方向未知，**修复前的数字不应作为最终论文数字**。建议再加一个“业务延迟”鲁棒性实验（例如 ERA5 约 5 天延迟：`valid_time + 5 day <= current_init`）。

## R2（高）SphereDyn + SphereTTC 的增益需要拆分来源

B 使用的是 SphereTTC 的多提供方约束组合模式（v22，`scripts/spheredyn/run_main_spherettc_v2.py` + `evaluate_goal_reference_ensemble.py`），与 proposal 中 “constrained combinations with trusted reference systems while preserving the contribution of the target backbone” 一致。实际做法是：

- 对每个（lead, 变量），SphereDyn 的权重固定为 0.5 或 0.75（或保持 Raw），其余权重分给 **FourCastNetV2、OneForecast、FuXi、Pangu、GraphCast** 五个参考预报，再加全局 bias 和空间 bias；
- 权重只用 2017 拟合并冻结，2018–2019 测试期间**不做在线更新**，也没有球谐变换；
- 脚本 docstring 写的是 “single-backbone SphereTTC-v22”，这个说法不准确，因为它用了 5 个外部预报。

Raw SphereDyn 的 macro nRMSE 是 0.3809，GraphCast Raw 是 0.2917，组合后是 0.2846。审稿人一定会问：这 25.28% 里有多少来自 SphereDyn、多少来自参考预报？`EXPERIMENT_SUMMARY_ZH.md` 第 6 节也写了“不能把最终结果归因于纯 SphereDyn”。

**需要补的对照**（都只需要已有缓存，计算量很小）：

1. SphereDyn + 单 backbone SphereTTC（`run_ttc.py --method sphere_ttc`，按 2017 协议选参）：说明球面校准本身对 SphereDyn 的作用；
2. 只用 5 个参考预报的同一种约束组合（不含 SphereDyn）：如果它明显好于 0.2846，就说明 SphereDyn 在组合里的贡献有限，需要调整叙述；
3. 把 SphereDyn 换成其他 backbone 做同样的组合（`evaluate_goal_reference_ensemble.py` 本来就支持 7 个 backbone）：说明“SphereDyn 是更好的组合成员”；
4. 如果坚持 “test-time” 的说法，可以加一个在 2018–2019 中按月滚动重拟合权重的在线版本。

## R3（高）SphereDyn 线不是盲测

- `PROTOCOL_V2.json`：“Existing 2018--2019 results remain development evidence and may support configuration selection, but they are not relabeled as blind evidence.”
- `spheredyn_v9_h100_paired_screen/DESIGN.json`：`evidence_class = 2018_DEVELOPMENT_...`，v9 是在 2018 上筛选晋级的；主运行的 2018 预报直接复用了这次筛选的缓存（`run_main_prediction_v2_4gpu.sh` 的 `EXISTING_2018`）。
- `EXPERIMENT_STATE_V2.json` 记录了 v22 在 2018–2019 的开发期结果（29.64%）。
- 协议在 2026-08-03 因“目标调整”从 v1 改为 v2：去掉了 GraphCast 硬门槛、多 seed、消融等要求；可选的 2023 prospective 从未运行。

**建议。** SphereDyn + SphereTTC 作为主结果时，最好补一次在没用过的年份（2020 或 2023）上的冻结评估（checkpoint 与 v22 权重都已冻结在本包里，只需推理）。补上之前，论文里要如实说明 2018–2019 参与过 SphereDyn 的开发筛选。

## R4（中）GraphCast 线 2018–2019 的测试集复用

时间线是：07-26 publication 设置在 2018–2019 上评估 → 08-04 parameter-only 搜索在 2018–2019 上 FAIL → 同日扩大搜索空间（lmax 到 64），选出 `l64_s08`。选择本身只用了 2017，但“扩大搜索空间”这个决定是在看过 2018–2019 的结果之后做的。

**建议。** GraphCast 一行照常放在 2018–2019 主表里，同时把 2020 prospective 作为最干净的稳健性证据，并在附录里写清这段参数选择历史。

## R5（中）FCNv2 / FuXi 的增益集中在由 RH 换算的比湿

| 模型 | q 的 Raw nRMSE | q 上的 SphereTTC 增益 | 其他变量族的增益 |
|---|---:|---:|---|
| FourCastNetV2 | 0.935 | 28.6% | z 9.2%，t 8.1%，u/v 约 3.6–3.8% |
| FuXi | 0.630 | 18.8% | z/t 约 2.9%，u/v 约 0.9–1.6% |

两个 adapter 都通过 `src/data_sources/fourcastnetv2_official.py::_specific_humidity_from_rh_temperature` 把相对湿度换算成 q。FCNv2 Raw 的 macro nRMSE（0.450）甚至比本地训练的 FNO（0.443）还差，这很可疑。审稿人会认为 SphereTTC 是在修 adapter 的系统偏差。

**建议。** 先核对换算：饱和水汽压公式、RH 是百分比还是 0–1、气压层单位。修复后重跑；或者在论文里把 q 单独报告，并给出不含 q 的 macro 指标。

## R6（中）ACC 与超参协议不一致

publication 设置（lmax 24，strength 1.0）在三个最强 backbone 上降低了 macro ACC：GraphCast 0.798825→0.796771，Pangu 0.788941→0.788252，OneForecast 0.773969→0.773594。`l64_s08` 只更新了 GraphCast 一行，所以现在主表里 GraphCast 和其余 10 个模型用的是两套不同的选择协议。

**建议。** 用同一个 2017 协议（两窗口、ACC 不下降约束）给 11 个 backbone 统一重选参数，或者统一使用 `l64_s08`，让主表口径一致。

## R7（中）对照、消融与随机性

- `run_ttc.py` 已经实现了 `static_bias`、`ema_bias`、`affine_ttc`、`geo_ttc`、`zonal_spectral_ttc`、`st_ttc` 等对照方法（`src/ttc/comparators/`），但本包里没有任何冻结的对照结果。顶会一定会要求和简单偏差校正（EMA bias / 线性 MOS）以及已有 TTC 方法比较。
- 没有消融：SHT 截断、chronological gate、robust 权重、taper、strength 各自的贡献都没有量化。`schedule/VALIDATION_CANDIDATES_2017.csv` 的 56 个候选可以先做一个超参敏感性图。
- SphereTTC 本身是确定性的；但 SphereDyn 只有 seed 44。

## R8（低）复现缺口

- 本包不含原始 S2S/ERA5 数据、逐 initialization 指标、预测缓存、GraphCast checkpoint 和官方模型源码（原服务器路径见 `02_REPRODUCE.md`）。
- SphereDyn 只有推理代码（`scripts/spheredyn/infer_spatiotemporal_checkpoint.py`），没有训练脚本，无法从头训练 v9。最后一阶段的训练配方记录在 checkpoint 的 `args` 里（见 `03_CODE_MAP.md` §3），但它是从 v8 checkpoint 热启动的，v8 及更早的训练链不在本包里，可能还在原服务器的 `SOON_CLEANUP_ARCHIVE_20260803` 中。
- 原 `requirements.txt` 漏掉了 `torch_harmonics`，而 SphereTTC（现 `src/spherettc.py`）依赖它。整理后的 `requirements.txt` 已补上，但当时的版本没有记录（单元测试在 0.8.0 下通过）。
- `scripts/spheredyn/evaluate_goal_reference_ensemble.py`（SOTA B 的 v22 权重选择逻辑）和 2026-08-04 记录的运行时版本除了路径引导以外还有其他差异，无法逐字节对齐；而且 SOTA B 是 08-03 跑的，早于这份哈希记录。好在拟合好的权重 `FITTED_WEIGHTS_2017.npz` 已冻结在本包里，应用阶段的代码（`run_main_spherettc_v2.py`，已对齐）只是一个线性组合。现在脚本总是重新拟合，建议加一个“加载已有权重”的选项，在原服务器上用这份权重重放一次 2018–2019，确认能复现 0.2845782（见 `03_CODE_MAP.md` §4）。
- 2018–2019 的 GraphCast 缓存在 V100 上生成，2020 的在 A100 上生成。

## R9（低）超参在网格边界

`l64_*` 是搜索里最大的截断阶数，`l64_s10` 的 RMSE 更好但 ACC 下降。建议补 lmax ∈ {80, 96, 120} 的 2017 搜索，并画敏感性曲线（也可以算作消融）。

## R10（低）评估口径

结果基于 1.5° 网格、UTC 日均值、5 个 lead 和 common-49 变量，backbone 是 GraphCast-small 1°（不是 0.25° 的业务版 GraphCast）。这和 WeatherBench2 的 6 小时瞬时 headline 不能直接比较，论文里要写明。

## R11（中）Proposal 描述与最终实现不完全一致

- proposal 写了 “Multi-timescale and seasonal calibration experts capture complementary short-term, persistent, and periodic error structures”。对应代码是 `src/ttc/legacy/sphere_v4.py`–`sphere_v7.py`（`run_ttc.py --method sphere_ttc_v4..v7`），但本包里**没有任何冻结结果使用它们**；v8–v21 的产物在 2026-08-03 清理时已移出，当时是否优于最终版无法从本包确认。
- 结果 B 使用的 v22 没有球谐分量，所以“SphereDyn + SphereTTC”里的 SphereTTC 和结果 A 里的 SphereTTC 不是同一个算子。

**建议。** 二选一：（a）论文的方法部分只保留最终结果真正用到的组件，把多时间尺度 / 季节专家放到讨论或未来工作；（b）用 2017 协议给 v4–v7 选参，在 11 个 backbone 上补一组结果，作为消融（“加入多时间尺度 / 季节专家的效果”）。同时在方法部分把单 backbone 模式（A）与多提供方模式（B）写成同一框架下的两种配置。

## R12（中）“first” 与 “SOTA” 的论断范围

- **first**：ST-TTC（NeurIPS 2025）提出了时空预报的 test-time computing，但实验只覆盖交通、空气质量、能源数据，没有全球中期天气，因此“首次系统研究 AI 天气模型的全球中期 test-time calibration”可以成立。但是业务数值预报很早就有基于延迟观测的在线偏差订正，例如 NCEP NAEFS 的 decaying-average bias correction、Kalman 滤波后处理和 MOS。审稿人如果来自气象背景，一定会提这些。Related work 必须讨论它们，实验里至少要和 `ema_bias`（即 decaying-average）以及 `static_bias` 对比。
- **SOTA**：Raw SphereDyn 在 5 个 lead、两个指标上都优于 6 个同协议 baseline，但平均水平不如 4 个官方发布模型。建议像 PnP-Corrector 那样，把主表分成“同协议训练”和“官方发布模型（冻结使用）”两组，只在前一组里称 SphereDyn 为 SOTA。

---

## 已核实没有问题的点

- 整理前后，只读验证器都能从冻结 CSV 逐字节重建全部 49 张表。
- `scripts/run_ttc.py` 和 SphereTTC 核心（原 `src/ttc/sphere.py`，现已原样并入 `src/spherettc.py`）：本仓库版本与 2026-08-04 产出 `l64_s08` 结果时的版本，除了 `sys.path` 引导和本次整理改的 import 路径以外完全一致（逐字节反推后 SHA-256 与 `schedule/integrity/source_before.json` 吻合，`verify_migration.py` 会检查）。
- publication 和 `l64_s08` 的参数都只用 2017 选择（`FROZEN_PARAMETERS.json`：`test_years_touched: []`；`FROZEN_PRIMARY_GLOBAL_2017.json`：冻结早于 2020 指标）。
- ACC climatology 只用 1979–2016 构建。
- SphereTTC-v22 的组合权重只用 2017 拟合（`FROZEN_RUN_DESIGN.json`：`fit_period = 2017-01-01/2017-12-31`），2018–2019 只做前向应用；SphereDyn checkpoint 哈希在运行前冻结。
- memory 只按 `valid_time` 过滤，不会用到 *valid_time > init* 的样本（除 R1 所说的日内部分之外）。

## 建议的补实验优先级

1. 修复 R1，并在 2017 / 2018–2019 / 2020 上重跑 5 个官方 backbone 的 SphereTTC（成本最低、最关键）。
2. 为结果 B 补 R2 中的拆分对照（SphereDyn + 单 backbone SphereTTC；只用参考预报的组合）。
3. 在没用过的年份（2020 或 2023）上冻结评估 SphereDyn 与 SphereDyn + SphereTTC（R3）。
4. 补 TTC / 偏差校正 baseline（尤其 `ema_bias`、`static_bias`、ST-TTC）与 SphereTTC 有/无球谐的消融（R7、R11、R12）；用统一的 2017 协议给 11 个 backbone 重选参数（R6、R9）。
5. 核对 RH→q 换算（R5）。
