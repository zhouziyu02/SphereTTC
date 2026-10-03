# 主实验复现说明

本工程以 `MAIN_RESULTS_49_VARIABLES_ZH.md` 为唯一主结果基准。清理后保留两级复现能力：

1. 表格级复现：由冻结的 11-baseline 与 SphereDyn 逐 initialization 指标重新聚合两份 CSV，并逐字节重建 49 张 Markdown 表；不训练、不推理、不需要 GPU。
2. 预测级追溯：保留 SphereDyn-v9 seed 44 checkpoint、2017–2019 SphereDyn 预测缓存、SphereTTC-v22 最终预测缓存、五个参考系统的 2017–2019 预测缓存、1979–2016 ACC climatology、标准化统计和主运行协议。

主预测 launcher 会优先使用当时的 `/tmp/spheredyn_development_daily_cache_1979_2022`（若存在）；该临时缓存不是复现前置条件，缺失时会直接读取工程内冻结的 `data/S2S`。原始运行 manifest 中的 `/tmp` 路径仅用于如实记录当时执行环境。

运行下列命令验证所有冻结文件 SHA-256、重新聚合两份 CSV，并确认 Markdown 与基准文件逐字节相同：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_main_results_reproducibility.py
```

如需把验证过程中重建的 Markdown 写到另一个位置：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_main_results_reproducibility.py --output /tmp/MAIN_RESULTS_49_VARIABLES_ZH.md
```

权威校验清单为 `MAIN_REPRODUCIBILITY_MANIFEST.json`。该验证只做确定性统计聚合，不开展新实验。
