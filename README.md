# SphereTTC

SphereTTC 是面向全球中期天气预报的单 backbone test-time calibrator。它保持预报模型冻结，在截断球谐空间中用历史预报–真值对拟合保守校准，并通过 chronological validation gate 控制校正强度。

本仓库包含核心实现、11 个 backbone 的实验入口、冻结参数与指标、对照方法和复现审计。当前主表为 **11 个 backbone × Raw / SphereTTC × 49 个变量 × 5 个时效 = 5,390 行**。

**已知限制：** 官方模型的日均 truth 在历史实验中可能过早进入 memory；相关数字（含 GraphCast 2020）在修复并重跑前仍为暂定。清理代码库未修复该算法问题。见 [风险审计](docs/04_AUDIT_AND_RISKS.md)。

## 从这里开始

- [项目概览](OVERVIEW.md)：方法、协议、结果范围与证据边界。
- [冻结结果](docs/01_SOTA_RESULT.md) / [论文表格](results/SPHERETTC_11_BACKBONES.md)。
- [复现指南](docs/02_REPRODUCE.md)：只读验证、测试、缓存重放和外部资产。
- [代码地图](docs/03_CODE_MAP.md) / [框架图说明](visualization.md)。
- [后续实验与论文叙事](docs/05_PAPER_STORYLINE.md)。

## 安装与测试

推荐 Python 3.11。下面的环境用于 SphereTTC 与 CPU 单元测试；真实全量实验默认使用 CUDA。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
python scripts/run_ttc.py --help
```

只需检查冻结表格与完整性时，可以在单独环境中仅安装 numpy：

```bash
python -m pip install -r env/requirements-verify.txt
PYTHONDONTWRITEBYTECODE=1 python verify_migration.py
```

验证器只读检查清单、SphereTTC 源码等价性及冻结结果的汇总一致性。它不能替代真实数据重跑，也不能证明实验不存在方法学问题。

## 主入口

核心模块：[`src/spherettc.py`](src/spherettc.py)。统一实验入口：[`scripts/run_ttc.py`](scripts/run_ttc.py)。

下面以 GraphCast `l64_s08` 为例；`/path/to/...` 需替换为已有缓存、truth 和 climatology 的路径，这些大体积资产不随仓库提供。命令重放历史协议，R1 修复后需要重新选参与评估。

```bash
mkdir -p runs
python scripts/run_ttc.py \
  --config configs/unified_11model_ttc.yaml --model graphcast \
  --cache /path/to/forecast_cache_2018_121x240.zarr \
  --truth-root /path/to/S2S \
  --acc-climatology /path/to/DAILY_DOY_1979_2016_COMMON49.npz \
  --method sphere_ttc \
  --params-json artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json \
  --warmup-cache /path/to/forecast_cache_2017_121x240.zarr \
  --trim-warmup-cache --warmup-truth-root /path/to/S2S \
  --no-cache-output --metrics-output runs/l64_s08_2018.npz \
  --profile-output runs/l64_s08_2018.json
```

其余 backbone 使用 `artifacts/ttc_publication_corrected_20260726/params/<model>.json`。Raw 对照使用 `--method raw`。完整路径与运行注意事项见 [复现指南](docs/02_REPRODUCE.md)。
