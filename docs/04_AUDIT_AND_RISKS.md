# 04 · SphereTTC 投稿前风险审计

本审计只针对单 backbone SphereTTC 在 11 个 backbone 上的冻结证据。**R1 必须修复并重跑，当前五个官方模型的结果（含 GraphCast 2020）仍为暂定。** 目录清理不改变算法、冻结数值或这些证据限制。为保持既有 TTC 记录的交叉引用，下文保留非连续的历史风险编号。

| 编号 | 严重度 | 风险 |
|---|---|---|
| R1 | 高 | 日均 truth 最多提前 18 h 进入 memory；影响五个官方 backbone |
| R4 | 中 | GraphCast 扩大搜索空间前已经查看 2018–2019 测试结果 |
| R5 | 中 | FourCastNetV2 / FuXi 大增益集中在由 RH 换算的 q |
| R6 | 中 | ACC 例外与不同 backbone 的选参协议差异 |
| R7 | 中 | 缺冻结 TTC 对照、组件消融和充分不确定性评估 |
| R8 | 低 | 原数据、缓存、官方代码 / checkpoint 与部分环境记录不在包内 |
| R9 | 低 | 选中的 lmax 64 处于搜索网格上界 |
| R10 | 低 | 日均 1.5°、GraphCast-small 口径与外部 headline 不可直接比较 |
| R11 | 中 | 多时间尺度 / 季节专家只有实验代码，无主结果证据 |
| R12 | 中 | “首次”“SOTA”“严格因果”等表述尚缺必要验证 |

## R1（高）日均目标的 delayed memory 最多前视 18 小时

**现象。** 官方模型（FourCastNetV2、OneForecast、FuXi、Pangu、GraphCast）在每天 00 UTC 初始化；lead *L* 的预报值是 *init+L* 当天 00/06/12/18 UTC 四个时刻的平均，truth 是同一 UTC 日的日均值（`configs/official_daily_common49_2018_2019.yaml`：`initialization_hour_utc: 0`、`daily_mean_sample_hours_utc: [0, 6, 12, 18]`）。但 memory 的准入条件是 `valid_time <= current_init`：

- `src/spherettc.py::eligible_from_buffer`：`np.datetime64(item[0]) <= current`（`scripts/run_ttc.py` 的所有在线方法都从这里导入）
- `src/spherettc.py::EligibleMemory`：`valid_times[:, lead] <= current`
- `tests/test_no_leakage_memory.py::test_valid_time_equal_current_init_is_eligible` 还把这条规则写成了测试

于是在 *D* 日 00 UTC 校准时，每个 lead 都会用到一对 truth 为 *D* 日日均值的样本，而这个日均值包含 *D* 日 06/12/18 UTC 的观测，相对当前初始化时刻是未来 6–18 小时的信息。这一对正好是 recency 权重最大的一条，也落在 chronological gate 的 holdout 段里。

**本地模型的时间语义。** 六个本地 backbone 的输入是 S2S 日均场，只有明确将 *D* 日预报发布时刻定义为 *D* 日结束，才能据此判断真值已经可用；不能仅凭使用日均输入就宣称无泄漏。论文与缓存时间坐标必须一致。

**修复。** 对日均目标，准入条件改为 `valid_time + 1 day <= current_init`（逐日 cadence 下等价于严格 `<`）。改 `src/spherettc.py` 里的 `eligible_from_buffer` 和 `EligibleMemory`（2026-10-03 合并后这是唯一位置），并更新 `tests/test_no_leakage_memory.py`。然后用原服务器上已有的预报缓存重跑：

1. GraphCast `l64_s08` 的 2017 选择（56 个候选，`artifacts/graphcast_spherettc_20260804/schedule/tools/search_schedule.py`）；
2. 2018–2019 回看（`run_retrospective_primary.py`）和 2020 prospective（`run_prospective_holdout.py`）；
3. 其余 10 个 backbone 的 SphereTTC（至少 4 个官方模型）。

单个 backbone 一年的 SphereTTC 在 A100 上约 2 分钟（`profiles/retrospective_l64_s08_2018.json`：125.7 s），整套重跑的成本很低。影响大小和方向尚未通过重跑确认，**修复前的数字不应作为最终论文数字**。建议再加一个“业务延迟”鲁棒性实验（例如额外延迟 5 天；应根据目标观测产品核对真实发布时间，并明确它与日均结束时刻的关系）。

## R4（中）GraphCast 线 2018–2019 的测试集复用

时间线是：07-26 publication 设置在 2018–2019 上评估 → 08-04 parameter-only 搜索在 2018–2019 上 FAIL → 同日扩大搜索空间（lmax 到 64），选出 `l64_s08`。选择本身只用了 2017，但“扩大搜索空间”这个决定是在看过 2018–2019 的结果之后做的。

**建议。** GraphCast 一行照常放在 2018–2019 主表里，同时把 2020 prospective 作为最干净的稳健性证据，并在附录里写清这段参数选择历史。2020 同样受 R1 影响；独立年份不等于其 causal memory 实现已经正确。

## R5（中）FCNv2 / FuXi 的增益集中在由 RH 换算的比湿

| 模型 | q 的 Raw nRMSE | q 上的 SphereTTC 增益 | 其他变量族的增益 |
|---|---:|---:|---|
| FourCastNetV2 | 0.935 | 28.6% | z 9.2%，t 8.1%，u/v 约 3.6–3.8% |
| FuXi | 0.630 | 18.8% | z/t 约 2.9%，u/v 约 0.9–1.6% |

两个 adapter 都通过 `src/data_sources/fourcastnetv2_official.py::_specific_humidity_from_rh_temperature` 把相对湿度换算成 q。FCNv2 Raw 的 macro nRMSE（0.450）高于本地训练的 FNO（0.443）。这提示需要排查 adapter 的系统偏差；仅凭这一排序不能断言换算实现错误。

**建议。** 先核对换算：饱和水汽压公式、RH 是百分比还是 0–1、气压层单位。修复后重跑；或者在论文里把 q 单独报告，并给出不含 q 的 macro 指标。

## R6（中）ACC 与超参协议不一致

publication 设置（lmax 24，strength 1.0）在三个最强 backbone 上降低了 macro ACC：GraphCast 0.798825→0.796771，Pangu 0.788941→0.788252，OneForecast 0.773969→0.773594。`l64_s08` 只更新了 GraphCast 一行，所以现在主表里 GraphCast 和其余 10 个模型用的是两套不同的选择协议。

**建议。** 用同一个 2017 协议（两窗口、ACC 不下降约束）给 11 个 backbone 统一重选参数，或在统一的 2017 验证上检验共享配置；不能未经验证就把 GraphCast 的 `l64_s08` 迁移到所有模型。

## R7（中）对照、消融与随机性

- `run_ttc.py` 已经实现了 `static_bias`、`ema_bias`、`affine_ttc`、`geo_ttc`、`zonal_spectral_ttc`、`st_ttc` 等对照方法（`src/ttc/comparators/`），但本包里没有任何冻结的对照结果。需要和简单偏差校正（EMA bias / 线性 MOS）以及已有 TTC 方法比较。
- 没有消融：SHT 截断、chronological gate、robust 权重、taper、strength 各自的贡献都没有量化。`schedule/VALIDATION_CANDIDATES_2017.csv` 的 56 个候选可以先做一个超参敏感性图。
- SphereTTC 本身是确定性的，但本地 backbone 的训练随机性、时间序列相关性和数据分布变化仍需单独评估；不能把确定性拟合当作结果无不确定性的依据。

## R8（低）复现缺口

- 本包不含原始 S2S/ERA5、逐 initialization 指标、预测缓存、官方模型源码或 checkpoint。原服务器资产路径见 `02_REPRODUCE.md`。本地 backbone 的预测缓存 / checkpoint 也需外部获取，当前包不保证从头训练全部模型。
- 原依赖列表遗漏 `torch_harmonics`，已补入当前 `requirements.txt`；历史运行版本未记录，历史单元测试在 0.8.0 下通过。
- GraphCast 2018–2019 缓存在 V100 生成，2020 缓存在 A100 生成，报告中须披露硬件差异。
- 当前 `verify_migration.py` 检查保留文件、表格和 SphereTTC 等价性证据；它没有执行真实预测或从原始数据重算指标。通过完整性验证不能填补这些缺口。

## R9（低）超参在网格边界

`l64_*` 是搜索里最大的截断阶数，`l64_s10` 的 RMSE 更好但 ACC 下降。建议补 lmax ∈ {80, 96, 120} 的 2017 搜索，并画敏感性曲线（也可以算作消融）。

## R10（低）评估口径

结果基于 1.5° 网格、UTC 日均值、5 个 lead 和 common-49 变量，backbone 是 GraphCast-small 1°（不是 0.25° 的业务版 GraphCast）。这和 WeatherBench2 的 6 小时瞬时 headline 不能直接比较，论文里要写明。

## R11（中）实验变体与主方法的边界

`src/ttc/legacy/sphere_v4.py`–`sphere_v7.py` 包含多时间尺度 / 季节残差专家，但没有冻结主结果使用这些组件。方法部分应只写 `src/spherettc.py` 实际用于 11 backbone 主结果的球谐仿射校准，把变体留作未来工作；若要声称其有效，需先在 2017 选参并做受控评估。

## R12（中）新颖性与优越性表述

现有结果仅比较 Raw 与 SphereTTC，不能支持“优于其他 TTC / 偏差校正方法”的 SOTA 结论。Related work 需要核查既有时空 TTC、业务 NWP 在线偏差订正（如 decaying-average）、Kalman 后处理和 MOS，并设置相应对照。文献核对完成前不使用“首次”的确定表述。

“无梯度训练”准确，“完全不拟合任何参数”不准确：SphereTTC 会在线拟合仿射校准系数。“严格因果”只能在 R1 修复、观测可用时刻明确并重跑之后用于相应结果。11/11 macro nRMSE 改善不能扩展成所有变量 / 时效改善，也不能扩展成 ACC 一致改善。

## 已保留的正面证据及其边界

- publication 和 `l64_s08` 的参数候选数值选择使用 2017；`FROZEN_PARAMETERS.json` 记录 `test_years_touched: []`。这不抹去 R4 的搜索空间开发历史。
- `FROZEN_PRIMARY_GLOBAL_2017.json` 的参数冻结早于 2020 指标；2020 是后续独立年份证据，但 R1 同样适用。
- ACC climatology 使用 1979–2016，normalization statistics 保存在 `artifacts/shared/s2s_daily_54var_stats.json`。
- memory 不会接纳标记为 `valid_time > init` 的样本；这只证明时间标签过滤，不能证明日均观测实际已可用。
- 保留的合并前 SphereTTC 源码、测试和 frozen CSV 支持实现等价性与结果汇总核对，不能替代原数据再运行。

## 补实验顺序

1. 修复 R1，同时更新 `eligible_from_buffer`、`EligibleMemory` 和边界测试；在 2017 / 2018–2019 / 2020 重跑受影响实验，保存新旧协议的差异。
2. 用同一可用真值规则和 2017 选参协议运行 static / EMA bias、grid affine 和已有 TTC 对照。
3. 补 SHT、gate、robust / recency 权重、taper 的消融，并统一 11 backbone 的超参选择；扩展 lmax 的敏感性评估。
4. 核对 RH→q 换算，补不含 q 的 macro 指标和分变量族分析。
5. 披露 GraphCast 的开发历史，保持未来年份冻结评估；报告逐单元回退、ACC 例外与适当的时间块不确定性。
