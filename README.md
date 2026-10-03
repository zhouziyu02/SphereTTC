# SphereCast：SphereTTC 测试时校准 + SphereDyn 球面动力学预报

> **Co-authors: please start with [`OVERVIEW.md`](OVERVIEW.md)（英文总览：方法、代码完整性、实验结果、待办）。**
> **代码：SphereTTC 全部在 `src/spherettc.py`，SphereDyn 全部在 `src/spheredyn.py`；画框架图请看 [`yaozhong.md`](yaozhong.md)。**

- **SphereTTC**：冻结任意天气预报 backbone，只用**已经核验过的**历史“预报–真值”对，在截断球谐空间里做保守的仿射校准；有多个预报提供方时，还能与可信参考系统做约束组合。
- **SphereDyn**：基于同样球面思想的预报 backbone（球面局部/谱动力学、semi-Lagrangian 输运、谐波气候趋势、多尺度球面残差），与 SphereTTC 组成端到端系统。

**定位**：test-time calibration / test-time computing 在 medium-range 天气（气候）预报上的第一次系统尝试。论证结构参照 PnP-Corrector（ICML 2026），由两个互补的 SOTA 组成：

- **A**：SphereTTC 作为即插即用的 calibrator，能有效提升现有 baseline；
- **B**：SphereDyn + SphereTTC 说明球面机理对这个任务有用，同时 SphereDyn 本身是同协议下的 SOTA baseline。

论文叙事、贡献草稿和实验规划见 `docs/05_PAPER_STORYLINE.md`。项目背景见 `docs/PROJECT_PROPOSAL.md`。

> 本仓库于 2026-10-02 原地整理：迭代版本、出表脚本、旧文档已分门别类，`artifacts/` 下的冻结实验证据一个字节都没改。整理前的完整快照在 `archive/ORIGINAL_SNAPSHOT_20261002.tar.gz`。

## SOTA 结果（2018–2019，730 个逐日 initialization，49 变量 × 5 时效）

所有校准参数只用 2017 年选择并冻结。macro nRMSE 越低越好。

**A. SphereTTC 作为 calibrator：在 11 个已有模型上一致提升**（`src/spherettc.py`，`scripts/run_ttc.py --method sphere_ttc`）

| | 本地训练 baseline（6 个） | 官方发布模型（5 个） |
|---|---|---|
| macro nRMSE 下降 | ConvLSTM −0.96%，Transformer −1.05%，FNO −0.55%，ViT −0.27%，CirT −0.10%，ClimODE −1.34% | FourCastNetV2 −19.82%，OneForecast −4.06%，FuXi −9.80%，Pangu −3.70%，GraphCast −4.57% |

- **11/11 个 backbone 改善**，macro nRMSE 平均下降 4.20%；5 个时效上平均都下降（24h 下降 6.28%，240h 下降 4.54%）；9/11 个 backbone 的 macro ACC 上升。
- GraphCast + SphereTTC（0.2784）是全表 macro nRMSE 最低的一行；在独立的 2020 prospective holdout 上 0/245 个单元变差，平均改善 3.57%。

**B. SphereDyn + SphereTTC：球面机理 + 同协议 SOTA baseline**（SphereDyn-v9 seed 44 + SphereTTC-v22 多提供方约束组合）

| | SphereDyn Raw | + SphereTTC | 对比 |
|---|---:|---:|---|
| macro nRMSE | 0.3809 | **0.2846（−25.28%）** | 低于全部 Raw 模型（GraphCast 0.2917）；全表第 2 |
| macro ACC | 0.6172 | **0.8080** | **全表第 1** |
| 168h / 240h nRMSE | 0.4553 / 0.4801 | **0.3532 / 0.4231** | 两个时效的 nRMSE 和 ACC 都是**全表第 1** |

- 245/245 个单元的 RMSE 和 ACC 全部提升。
- Raw SphereDyn 在 5 个 lead、nRMSE 与 ACC 两个指标上都优于全部 6 个同协议训练的 baseline（最好的 FNO 为 0.4427）。官方发布模型单独成组，不在这一比较范围内。

详细说明见 `docs/01_SOTA_RESULT.md`。生成好的表格在 `results/`：论文表格草稿 `PAPER_TABLE1_NRMSE_BY_LEAD.md`、`PAPER_TABLE2_ACC_BY_LEAD.md`、`EFFICIENCY.md`，以及 `SPHERETTC_11_BACKBONES.md`、`SPHEREDYN_SPHERETTC.md`、`HEADLINE_120H/240H_LATEST.md`。

## ⚠️ 投稿前必须先处理

详见 `docs/04_AUDIT_AND_RISKS.md`，最重要的四条：

1. **R1**：日均目标进入 memory 的条件是 `valid_time <= init`，对 00 UTC 初始化的 5 个官方模型存在最多 18h 的日内前视。修复只需一行代码，但这 5 个模型的 SphereTTC 结果需要重跑（成本很低）。
2. **R2**：SphereDyn + SphereTTC 的 25.28% 包含与 5 个参考预报（含 GraphCast）的约束组合。论文需要补“SphereDyn + 单 backbone SphereTTC”和“只用参考预报的组合”两个对照，把增益来源拆开。
3. **R3**：SphereDyn 线在开发阶段用过 2018–2019，需要一个没用过的年份做冻结评估。
4. **R11**：proposal 写了 multi-timescale / seasonal experts，但最终结果没有用到（代码在 `src/ttc/legacy/`）。

## 目录结构

```
.
├── OVERVIEW.md                ← 英文总览（给 co-author）
├── yaozhong.md                ← 画 SphereTTC / SphereDyn 框架图的指南
├── README.md                  ← 中文入口
├── verify_migration.py        ← 一键只读验证（只需 numpy）
├── requirements.txt           ← SphereTTC 运行 + 单元测试依赖
├── MANIFEST.sha256            ← 全部文件的 SHA-256
├── docs/                      ← 01 SOTA 结果 / 02 复现 / 03 代码地图 / 04 审计与风险 / 05 论文叙事 / 项目提案
├── results/                   ← 由 tools/build_results_summary.py 生成：论文表格草稿、两个 SOTA 的表格、效率、全表排名
├── src/
│   ├── spherettc.py           ← ★ SphereTTC 完整模块（单文件：延迟记忆、球谐校准、在线插件、多提供方模式）
│   ├── spheredyn.py           ← ★ SphereDyn 完整模型（单文件）
│   ├── ttc/metrics.py         ← 评估指标
│   ├── ttc/comparators/       ← 对照方法：geo_ttc / zonal_spectral_ttc / ST-TTC
│   ├── ttc/legacy/            ← SphereTTC v2–v7（多频带 / 多时间尺度 / 季节专家，无冻结结果）
│   ├── baselines/             ← 6 个同协议 baseline + build_model
│   ├── data_sources/          ← 5 个官方模型 adapter + S2S 日均 truth
│   └── experiments/, utils/
├── scripts/
│   ├── run_ttc.py             ← ★ 所有 TTC 方法的统一入口（SOTA 入口）
│   ├── tables/                ← 指标汇总与 49 张表渲染
│   ├── spheredyn/             ← ★ SphereDyn 推理 + SphereTTC-v22（SOTA B）+ 指标 + 4-GPU 启动脚本
│   └── preprocessing/         ← ACC climatology
├── configs/                   ← 实验配置
├── tests/                     ← 单元测试（合成数据）
├── tools/                     ← build_results_summary.py
├── env/                       ← 其他环境：verify-only / GraphCast JAX GPU / 外部 baseline
├── artifacts/                 ← 冻结实验证据（只读，见 artifacts/README.md）
└── archive/                   ← 整理前快照、旧 README、溯源日志、只能在原服务器运行的验证器
```

## 快速开始

```bash
# L0：只读验证（不训练、不推理、不写文件）
python -m pip install -r env/requirements-verify.txt
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py
# 期望输出以 SPHERETTC_REPRODUCIBLE 开头，并包含 spherettc_backbones_improved=11/11

# L1：单元测试
python -m pip install -r requirements.txt
pytest -q
```

从冻结缓存重跑 SphereTTC、重新生成 GraphCast 预报、重跑 SphereDyn + SphereTTC 需要原服务器上的数据和 checkpoint，命令与资产清单见 `docs/02_REPRODUCE.md`。

## 文档

| 文档 | 内容 |
|---|---|
| `OVERVIEW.md` | 英文总览：两个结果、方法、协议、代码完整性核查、已知问题与待办 |
| `yaozhong.md` | 画框架图指南：对照 `src/spherettc.py` 和 `src/spheredyn.py` 的模块、数据流、张量形状与参数量 |
| `docs/01_SOTA_RESULT.md` | 两个 SOTA（最终版 SphereTTC、SphereDyn + SphereTTC）的定义、参数、逐模型/逐时效结果、与 proposal 的对照、论文措辞建议 |
| `docs/02_REPRODUCE.md` | L0–L5 复现层级、命令、所需外部资产与原服务器路径 |
| `docs/03_CODE_MAP.md` | 代码与结果的对应、SOTA 调用链、版本谱系、代码溯源校验、时间线 |
| `docs/04_AUDIT_AND_RISKS.md` | 从审稿人角度的 12 条风险、已核实没问题的点、补实验优先级 |
| `docs/05_PAPER_STORYLINE.md` | 参照 PnP-Corrector 的论文叙事、两个 SOTA 的分工、贡献草稿（英文）、实验章节规划与完成状态 |
| `artifacts/README.md` | 三组冻结实验的索引 |
| `archive/README.md` | 历史文件与整理溯源 |
