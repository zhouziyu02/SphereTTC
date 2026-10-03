#!/usr/bin/env python
"""Build the consolidated latest tables and GraphCast comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.tables.render_main_results_49_variables_zh import _load_rows, _render


ARTIFACT = REPOSITORY / "artifacts/graphcast_spherettc_20260804"
BASELINE_CSV = (
    REPOSITORY
    / "artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv"
)
SPHEREDYN_CSV = (
    REPOSITORY
    / "artifacts/spheredyn_spherettc_open_goal_20260801/"
    "main_prediction_v2_seed44/paper_metrics/SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv"
)
ORIGINAL_GRAPHCAST_METRICS = (
    REPOSITORY
    / "artifacts/ttc_publication_corrected_20260726/metrics/graphcast"
)
SCHEDULE = ARTIFACT / "schedule"
PARAMETER_ONLY = ARTIFACT / "parameter_only"
METRIC_FIELDS = (
    "mse",
    "mae",
    "bias",
    "calibration_gate",
    "acc_numerator",
    "acc_prediction_energy",
    "acc_target_energy",
)


def load_metric(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def concatenate(paths: list[Path]) -> dict[str, np.ndarray]:
    items = [load_metric(path) for path in paths]
    for coordinate in ("lead_time", "variable"):
        if not all(
            np.array_equal(items[0][coordinate], item[coordinate])
            for item in items[1:]
        ):
            raise RuntimeError(f"metric coordinate mismatch: {coordinate}")
    result = {
        "lead_time": items[0]["lead_time"],
        "variable": items[0]["variable"],
        "init_time": np.concatenate([item["init_time"] for item in items]),
    }
    for name in METRIC_FIELDS:
        if name in items[0]:
            result[name] = np.concatenate([item[name] for item in items], axis=0)
    return result


def aggregate(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rmse = np.sqrt(data["mse"].mean(axis=0))
    numerator = data["acc_numerator"].sum(axis=0)
    prediction = data["acc_prediction_energy"].sum(axis=0)
    target = data["acc_target_energy"].sum(axis=0)
    acc = numerator / np.sqrt(np.maximum(prediction * target, 1e-24))
    return rmse, acc


def paired_summary(
    raw: dict[str, np.ndarray], calibrated: dict[str, np.ndarray]
) -> dict[str, Any]:
    for coordinate in ("init_time", "lead_time", "variable"):
        if not np.array_equal(raw[coordinate], calibrated[coordinate]):
            raise RuntimeError(f"paired metric mismatch: {coordinate}")
    raw_rmse, raw_acc = aggregate(raw)
    calibrated_rmse, calibrated_acc = aggregate(calibrated)
    gain = 1.0 - calibrated_rmse / raw_rmse
    acc_delta = calibrated_acc - raw_acc
    by_lead: dict[str, dict[str, Any]] = {}
    for index, lead in enumerate(raw["lead_time"]):
        by_lead[str(int(lead))] = {
            "mean_relative_rmse_gain": float(gain[index].mean()),
            "negative_rmse_cells": int((gain[index] < 0).sum()),
            "mean_acc_delta": float(acc_delta[index].mean()),
            "negative_acc_cells": int((acc_delta[index] < 0).sum()),
        }
    return {
        "initializations": int(len(raw["init_time"])),
        "cell_count": int(gain.size),
        "mean_relative_rmse_gain": float(gain.mean()),
        "median_relative_rmse_gain": float(np.median(gain)),
        "minimum_relative_rmse_gain": float(gain.min()),
        "negative_rmse_cells": int((gain < 0).sum()),
        "negative_rmse_cell_fraction": float((gain < 0).mean()),
        "mean_acc_delta": float(acc_delta.mean()),
        "minimum_acc_delta": float(acc_delta.min()),
        "negative_acc_cells": int((acc_delta < 0).sum()),
        "by_lead": by_lead,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_graphcast_metrics() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    root = SCHEDULE / "metrics/retrospective_2018_2019"
    raw = concatenate([root / "raw_2018.npz", root / "raw_2019.npz"])
    calibrated = concatenate(
        [root / "l64_s08_2018.npz", root / "l64_s08_2019.npz"]
    )
    return raw, calibrated


def replace_graphcast_rows(
    baseline_rows: list[dict[str, str]], calibrated: dict[str, np.ndarray]
) -> list[dict[str, str]]:
    rmse, acc = aggregate(calibrated)
    leads = {int(value): index for index, value in enumerate(calibrated["lead_time"])}
    variables = {
        str(value): index for index, value in enumerate(calibrated["variable"])
    }
    result: list[dict[str, str]] = []
    replaced = 0
    for original in baseline_rows:
        row = dict(original)
        if row["model"] == "graphcast" and row["method"] == "sphere_ttc":
            lead_index = leads[int(row["lead_time_hours"])]
            variable_index = variables[row["variable"]]
            row["rmse"] = str(float(rmse[lead_index, variable_index]))
            row["acc"] = str(float(acc[lead_index, variable_index]))
            row["n_initializations"] = str(len(calibrated["init_time"]))
            replaced += 1
        result.append(row)
    if replaced != 245:
        raise RuntimeError(f"expected to replace 245 GraphCast rows, replaced {replaced}")
    return result


def render_latest(latest_csv: Path) -> str:
    variables, units, values = _load_rows([latest_csv])
    base = _render(
        [latest_csv.relative_to(REPOSITORY) if latest_csv.is_relative_to(REPOSITORY) else latest_csv],
        variables,
        units,
        values,
    )
    tables = base[base.index("## 变量索引") :]
    header = """# 最新统一 49 变量完整实验表（GraphCast 全局参数更新）

> 数据状态：冻结的最新综合结果  
> 评估时期：2018-01-01 至 2019-12-31  
> Initialization：730  
> 模型：原 11 个 baseline + SphereDyn-v9 seed 44  
> 更新：仅将 GraphCast + SphereTTC 行替换为 2017 冻结的全局 `l64_s08` 结果  
> 变量：49；时效：24、72、120、168、240h

## 阅读说明

本表保留原 49 表的模型、变量、时效和测试期，仅更新后续实验明确改善的 GraphCast + SphereTTC 行。其他 10 个 baseline、所有 Raw 行以及 SphereDyn-v9 / SphereTTC-v22 行保持原冻结结果不变，仍为 5,880 条结果、49 张表、每表 24 行。

GraphCast 全局参数 `l64_s08` 只用 2017 的搜索段与时间后置 holdout 冻结；2018–2019 仅作冻结后同口径回看。独立的 2020 prospective 结果不与其他模型跨年份混入本表，详见同目录 `schedule/PROSPECTIVE_2020_REPORT.md`。

RMSE 越低越好，ACC 越高越好；SphereTTC 行中相对同模型 Raw 改善的数值使用粗体。原始 49 表继续保留为不可变历史基准。

生成源：`artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`

"""
    return header + tables


def selected_summary(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "initializations",
        "scored_initializations",
        "cell_count",
        "mean_relative_rmse_gain",
        "median_relative_rmse_gain",
        "minimum_relative_rmse_gain",
        "negative_rmse_cells",
        "negative_rmse_cell_fraction",
        "mean_acc_delta",
        "minimum_acc_delta",
        "negative_acc_cells",
        "by_lead",
    )
    return {key: data[key] for key in keys if key in data}


def comparison_payload(
    latest_raw: dict[str, np.ndarray], latest_calibrated: dict[str, np.ndarray]
) -> dict[str, Any]:
    original_raw = load_metric(
        ORIGINAL_GRAPHCAST_METRICS / "raw/per_initialization_metrics.npz"
    )
    original_ttc = load_metric(
        ORIGINAL_GRAPHCAST_METRICS / "sphere_ttc/per_initialization_metrics.npz"
    )
    original = paired_summary(original_raw, original_ttc)
    latest = paired_summary(latest_raw, latest_calibrated)
    parameter = json.loads((PARAMETER_ONLY / "test_summary.json").read_text())
    prospective = json.loads(
        (SCHEDULE / "PROSPECTIVE_2020_SUMMARY.json").read_text()
    )
    return {
        "frozen_49_table_graphcast_spherettc_2018_2019": original,
        "parameter_only_l24_strength050_2018_2019": selected_summary(parameter),
        "latest_global_l64_s08_2018_2019": latest,
        "latest_global_l64_s08_2020_prospective": prospective["primary_global"],
        "secondary_family_lead_schedule_2020_prospective": prospective[
            "secondary_family_lead_schedule"
        ],
        "latest_vs_frozen_49_table": {
            "mean_rmse_gain_percentage_point_change": 100.0
            * (
                latest["mean_relative_rmse_gain"]
                - original["mean_relative_rmse_gain"]
            ),
            "mean_rmse_gain_relative_change": (
                latest["mean_relative_rmse_gain"]
                / original["mean_relative_rmse_gain"]
                - 1.0
            ),
            "macro_acc_delta_change": latest["mean_acc_delta"]
            - original["mean_acc_delta"],
            "acc_regressed_cells_change": latest["negative_acc_cells"]
            - original["negative_acc_cells"],
            "rmse_regressed_cells_change": latest["negative_rmse_cells"]
            - original["negative_rmse_cells"],
        },
    }


def render_summary(comparison: dict[str, Any]) -> str:
    old = comparison["frozen_49_table_graphcast_spherettc_2018_2019"]
    param = comparison["parameter_only_l24_strength050_2018_2019"]
    new = comparison["latest_global_l64_s08_2018_2019"]
    p2020 = comparison["latest_global_l64_s08_2020_prospective"]
    schedule = comparison["secondary_family_lead_schedule_2020_prospective"]
    return f"""# SOON 最新完整实验结果（截至 2026-08-04）

## 结论

原 49 表已达到 SphereDyn 主目标并保持为不可变历史基准。后续 GraphCast + SphereTTC 确有实验进展：推荐将单一全局 `l64_s08` 作为最新 GraphCast 主结果；family/lead schedule 只作为更灵活的次要分析。

| 实验 | 时期 | 平均单元 RMSE 改善 | RMSE 退化 | macro ACC 变化 | ACC 退化 | 判定 |
|---|---|---:|---:|---:|---:|---|
| 原 49 表 GraphCast + publication SphereTTC | 2018–2019 | {100 * old['mean_relative_rmse_gain']:.4f}% | {old['negative_rmse_cells']}/245 | {old['mean_acc_delta']:+.8f} | {old['negative_acc_cells']}/245 | 历史基准 |
| parameter-only `l24_strength050` | 2018–2019 | {100 * param['mean_relative_rmse_gain']:.4f}% | {param['negative_rmse_cells']}/245 | {param['mean_acc_delta']:+.8f} | {param['negative_acc_cells']}/245 | FAIL |
| 全局 `l64_s08` | 2018–2019 | {100 * new['mean_relative_rmse_gain']:.4f}% | {new['negative_rmse_cells']}/245 | {new['mean_acc_delta']:+.8f} | {new['negative_acc_cells']}/245 | 同口径改善 |
| 全局 `l64_s08` | 2020 prospective | {100 * p2020['mean_relative_rmse_gain']:.4f}% | {p2020['negative_rmse_cells']}/245 | {p2020['mean_acc_delta']:+.8f} | {p2020['negative_acc_cells']}/245 | PASS |
| family/lead schedule | 2020 prospective | {100 * schedule['mean_relative_rmse_gain']:.4f}% | {schedule['negative_rmse_cells']}/245 | {schedule['mean_acc_delta']:+.8f} | {schedule['negative_acc_cells']}/245 | PASS（次要） |

全局 `l64_s08` 相比原 49 表中的 GraphCast + SphereTTC，把同一 2018–2019 测试上的平均 RMSE 改善提高 {100 * (new['mean_relative_rmse_gain'] - old['mean_relative_rmse_gain']):.4f} 个百分点，并把 macro ACC 从负变化改为正变化；ACC 退化单元由 {old['negative_acc_cells']} 降至 {new['negative_acc_cells']}。

## 权威结果入口

- 原冻结 49 表：`../spheredyn_spherettc_open_goal_20260801/MAIN_RESULTS_49_VARIABLES_ZH.md`
- 更新 GraphCast 行后的最新 49 表：`LATEST_RESULTS_49_VARIABLES_ZH.md`
- 机器可读最新 5,880 行：`LATEST_RESULTS.csv`
- 2020 prospective 结果：`schedule/PROSPECTIVE_2020_REPORT.md`
- 完整数值对比：`COMPARISON_WITH_FROZEN_49_TABLES.json`

## 解释边界

- SphereDyn 主结果仍只有 seed 44，且 SphereTTC-v22 使用外部参考预测。
- `l64_s08` 是一个全局参数集，2017 冻结后才用于 2018–2019 与 2020；2020 是新增的独立 prospective 证据。
- family/lead schedule 自由度更高，因此不替代全局参数作为 headline。
- 2020 只新增 GraphCast 对照，不能把它与其他模型的 2018–2019 数值直接排名。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fieldnames, baseline_rows = read_csv(BASELINE_CSV)
    sphere_fields, sphere_rows = read_csv(SPHEREDYN_CSV)
    if sphere_fields != fieldnames:
        raise RuntimeError("baseline and SphereDyn CSV schemas differ")
    latest_raw, latest_calibrated = latest_graphcast_metrics()
    updated_baselines = replace_graphcast_rows(baseline_rows, latest_calibrated)
    latest_rows = [*updated_baselines, *sphere_rows]
    latest_csv = output / "LATEST_RESULTS.csv"
    write_csv(latest_csv, fieldnames, latest_rows)
    write_csv(
        output / "GRAPHCAST_L64_S08_2018_2019.csv",
        fieldnames,
        [row for row in latest_rows if row["model"] == "graphcast"],
    )
    (output / "LATEST_RESULTS_49_VARIABLES_ZH.md").write_text(
        render_latest(latest_csv), encoding="utf-8"
    )
    comparison = comparison_payload(latest_raw, latest_calibrated)
    (output / "COMPARISON_WITH_FROZEN_49_TABLES.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "LATEST_EXPERIMENT_SUMMARY_ZH.md").write_text(
        render_summary(comparison), encoding="utf-8"
    )
    print(
        "LATEST_RESULTS_BUILT rows=5880 variables=49 tables=49 "
        "graphcast_replacements=245"
    )


if __name__ == "__main__":
    main()
