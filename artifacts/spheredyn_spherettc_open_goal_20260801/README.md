# 冻结主实验

本目录只保留以 `MAIN_RESULTS_49_VARIABLES_ZH.md` 为基准的主实验结果与复现资料。

- `PROTOCOL_V2.json`：冻结实验协议。
- `EXPERIMENT_STATE_V2.json`：完成状态。
- `MAIN_RESULTS_49_VARIABLES_ZH.md`：49 变量、五时效 RMSE/ACC 权威表格。
- `EXPERIMENT_SUMMARY_ZH.md`：主实验设置、结果与解释边界。
- `MAIN_REPRODUCIBILITY_MANIFEST.json`：输入、指标与最终表格的 SHA-256。
- `REPRODUCIBILITY.md`：无训练、无推理的确定性复现命令。
- `CLEANUP_RECORD.md`：保留范围、移除范围与可恢复归档位置。
- `main_prediction_v2_seed44/`：SphereDyn-v9 seed 44 主预测、SphereTTC-v22 结果与逐样本指标。
- `spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/`：主实验唯一使用的 SphereDyn checkpoint 与 2018 预测缓存。

主结果：SphereDyn-v9 + SphereTTC-v22 的 macro nRMSE 为 0.2845782224，相比 Raw SphereDyn 的 0.3808743035 提升 25.2829%。
