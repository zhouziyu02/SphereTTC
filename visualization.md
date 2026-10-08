# SphereTTC 框架图

从 [`src/spherettc.py`](src/spherettc.py) 的模块说明、`SphereTTCCalibrator` 和在线插件 `SphereTTC` 开始读；实验入口是 [`scripts/run_ttc.py`](scripts/run_ttc.py)。主图只画用于冻结 11 backbone 结果的单 backbone 方法。

```text
Frozen backbone ─── current forecast ────────────────────────────┐
       │                                                       │
       └── past forecasts + available truths                    │
                    │                                          │
              delayed memory                                   │
                    │                                          │
            truncated spherical harmonics                      │
                    │                                          │
       recency / robust weighted affine fit                    │
          + shrinkage toward identity                          │
                    │                                          │
       chronological validation → reliability gate             │
                    │                                          │
          degree taper → inverse transform                     │
                    └──────── correction ────────────────────── +
                                                               │
                                                        calibrated forecast
```

- 表示每个 lead / variable 的独立拟合和 backbone 冻结；不要画反向传播或 backbone 更新。
- gate = 0 时沿原预报输出，作为 no-op 分支。
- 历史真值箭头应标注 **available after verification and observation delay**。当前实现按 `valid_time <= init` 准入，日均可用时刻问题尚待修复（[R1](docs/04_AUDIT_AND_RISKS.md)）；图注不能把现有结果直接称为严格因果。
- 图中不加入未产生冻结结果的多时间尺度或季节专家。方法边界与证据见 [结果说明](docs/01_SOTA_RESULT.md)。
