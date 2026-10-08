#!/usr/bin/env python3
"""Rebuild the SphereTTC tables from the portable frozen aggregate CSVs.

No per-initialization tensors, forecasting checkpoints, or GPU are required.
Scalar metric strings are copied verbatim; only GraphCast calibration rows are
replaced by the frozen l64_s08 extract. GraphCast comparison summaries remain
frozen evidence and are independently checked by verify_migration.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts.tables.render_main_results_49_variables_zh import _load_rows, _render

ARTIFACT = REPOSITORY / "artifacts/graphcast_spherettc_20260804"
BASELINE_CSV = REPOSITORY / "artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv"
GRAPHCAST_CSV = ARTIFACT / "GRAPHCAST_L64_S08_2018_2019.csv"
KEY = ("model", "method", "period", "lead_time_hours", "variable")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def latest_csv_text() -> str:
    fields, baseline = read_csv(BASELINE_CSV)
    graphcast_fields, graphcast = read_csv(GRAPHCAST_CSV)
    if fields != graphcast_fields or len(baseline) != 5390 or len(graphcast) != 490:
        raise ValueError("frozen CSV schema or row count mismatch")
    replacement = {tuple(row[k] for k in KEY): row for row in graphcast}
    if len(replacement) != 490 or any(row["model"] != "graphcast" for row in graphcast):
        raise ValueError("invalid GraphCast extract")
    rows = []
    replaced = 0
    for row in baseline:
        if row["model"] == "graphcast":
            update = replacement[tuple(row[k] for k in KEY)]
            for field in fields:
                if field not in {"rmse", "acc"} and row[field] != update[field]:
                    raise ValueError(f"GraphCast metadata changed: {field}")
            if row["method"] == "raw" and row != update:
                raise ValueError("frozen GraphCast Raw metrics changed")
            row = update
            replaced += row["method"] == "sphere_ttc"
        rows.append(row)
    if replaced != 245:
        raise ValueError("expected 245 calibrated GraphCast cells")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_latest(latest_csv: Path, *, csv_text: str | None = None) -> str:
    variables, units, values = _load_rows([latest_csv], csv_text=csv_text)
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
> 模型：11 个冻结 forecasting backbone
>
> 更新：仅将 GraphCast + SphereTTC 行替换为 2017 冻结的全局 `l64_s08` 结果  
> 变量：49；时效：24、72、120、168、240h

## 阅读说明

本表汇总 11 个 backbone 的 Raw 与 SphereTTC 结果：5,390 条结果、49 张表、每表 22 行。GraphCast + SphereTTC 使用后续冻结的全局参数；其他 10 个 backbone 及所有 Raw 行保持原冻结结果不变。

GraphCast 全局参数 `l64_s08` 只用 2017 的搜索段与时间后置 holdout 冻结；2018–2019 仅作冻结后同口径回看。独立的 2020 prospective 结果不与其他模型跨年份混入本表，详见同目录 `schedule/PROSPECTIVE_2020_REPORT.md`。

RMSE 越低越好，ACC 越高越好；SphereTTC 行中相对同模型 Raw 改善的数值使用粗体。publication 参数下的 11-backbone 结果保留在 `../ttc_publication_corrected_20260726/results/MAIN_RESULTS_49_VARIABLES_ZH.md`。

生成源：`artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`

"""
    return header + tables


def render_summary(comparison: dict[str, Any]) -> str:
    old = comparison["frozen_49_table_graphcast_spherettc_2018_2019"]
    param = comparison["parameter_only_l24_strength050_2018_2019"]
    new = comparison["latest_global_l64_s08_2018_2019"]
    p2020 = comparison["latest_global_l64_s08_2020_prospective"]
    schedule = comparison["secondary_family_lead_schedule_2020_prospective"]
    return f"""# SphereTTC 最新完整实验结果（截至 2026-08-04）

## 结论

11-backbone 的 publication 结果保留为历史基准。GraphCast + SphereTTC 的后续实验表明：推荐将单一全局 `l64_s08` 作为最新 GraphCast 主结果；family/lead schedule 只作为更灵活的次要分析。

| 实验 | 时期 | 平均单元 RMSE 改善 | RMSE 退化 | macro ACC 变化 | ACC 退化 | 判定 |
|---|---|---:|---:|---:|---:|---|
| 原 49 表 GraphCast + publication SphereTTC | 2018–2019 | {100 * old['mean_relative_rmse_gain']:.4f}% | {old['negative_rmse_cells']}/245 | {old['mean_acc_delta']:+.8f} | {old['negative_acc_cells']}/245 | 历史基准 |
| parameter-only `l24_strength050` | 2018–2019 | {100 * param['mean_relative_rmse_gain']:.4f}% | {param['negative_rmse_cells']}/245 | {param['mean_acc_delta']:+.8f} | {param['negative_acc_cells']}/245 | FAIL |
| 全局 `l64_s08` | 2018–2019 | {100 * new['mean_relative_rmse_gain']:.4f}% | {new['negative_rmse_cells']}/245 | {new['mean_acc_delta']:+.8f} | {new['negative_acc_cells']}/245 | 同口径改善 |
| 全局 `l64_s08` | 2020 prospective | {100 * p2020['mean_relative_rmse_gain']:.4f}% | {p2020['negative_rmse_cells']}/245 | {p2020['mean_acc_delta']:+.8f} | {p2020['negative_acc_cells']}/245 | PASS |
| family/lead schedule | 2020 prospective | {100 * schedule['mean_relative_rmse_gain']:.4f}% | {schedule['negative_rmse_cells']}/245 | {schedule['mean_acc_delta']:+.8f} | {schedule['negative_acc_cells']}/245 | PASS（次要） |

全局 `l64_s08` 相比原 49 表中的 GraphCast + SphereTTC，把同一 2018–2019 测试上的平均 RMSE 改善提高 {100 * (new['mean_relative_rmse_gain'] - old['mean_relative_rmse_gain']):.4f} 个百分点，并把 macro ACC 从负变化改为正变化；ACC 退化单元由 {old['negative_acc_cells']} 降至 {new['negative_acc_cells']}。

## 权威结果入口

- 原冻结 49 表：`../ttc_publication_corrected_20260726/results/MAIN_RESULTS_49_VARIABLES_ZH.md`
- 更新 GraphCast 行后的最新 49 表：`LATEST_RESULTS_49_VARIABLES_ZH.md`
- 机器可读最新 5,390 行：`LATEST_RESULTS.csv`
- 2020 prospective 结果：`schedule/PROSPECTIVE_2020_REPORT.md`
- 完整数值对比：`COMPARISON_WITH_FROZEN_49_TABLES.json`

## 解释边界

- `l64_s08` 是一个全局参数集，2017 冻结后才用于 2018–2019 与 2020；2020 是新增的独立 prospective 证据。
- family/lead schedule 自由度更高，因此不替代全局参数作为 headline。
- 2020 只新增 GraphCast 对照，不能把它与其他模型的 2018–2019 数值直接排名。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ARTIFACT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    comparison = json.loads((ARTIFACT / "COMPARISON_WITH_FROZEN_49_TABLES.json").read_text(encoding="utf-8"))
    csv_text = latest_csv_text()
    outputs = {
        "LATEST_RESULTS.csv": csv_text,
        "LATEST_RESULTS_49_VARIABLES_ZH.md": render_latest(ARTIFACT / "LATEST_RESULTS.csv", csv_text=csv_text),
        "LATEST_EXPERIMENT_SUMMARY_ZH.md": render_summary(comparison),
    }
    if args.check:
        bad = [name for name, text in outputs.items()
               if not (output / name).is_file() or (output / name).read_bytes() != text.encode("utf-8")]
        if bad:
            raise SystemExit(f"latest results out of date: {bad}")
        print("LATEST_RESULTS_UP_TO_DATE rows=5390 variables=49 tables=49 configurations=22")
        return
    output.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (output / name).write_bytes(text.encode("utf-8"))
    print("LATEST_RESULTS_BUILT rows=5390 variables=49 tables=49 configurations=22 graphcast_replacements=245")


if __name__ == "__main__":
    main()
