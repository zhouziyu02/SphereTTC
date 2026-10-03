#!/usr/bin/env python
"""Render the frozen unified 11-model results as 49 Chinese Markdown tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_ORDER = (
    "convlstm",
    "transformer",
    "fno",
    "vit",
    "cirt",
    "climode",
    "fourcastnetv2",
    "oneforecast",
    "fuxi",
    "pangu",
    "graphcast",
    "spheredyn_v9",
)
MODEL_NAMES = {
    "convlstm": "ConvLSTM",
    "transformer": "Transformer",
    "fno": "FNO",
    "vit": "ViT",
    "cirt": "CirT",
    "climode": "ClimODE-style adapter",
    "fourcastnetv2": "FourCastNetV2",
    "oneforecast": "OneForecast",
    "fuxi": "FuXi",
    "pangu": "Pangu",
    "graphcast": "GraphCast-small 1°",
    "spheredyn_v9": "SphereDyn-v9 (seed 44)",
}
LEADS = (24, 72, 120, 168, 240)
METHODS_BY_MODEL = {
    **{model: ("raw", "sphere_ttc") for model in MODEL_ORDER[:-1]},
    "spheredyn_v9": ("raw", "sphere_ttc_v22"),
}
PREFIX_NAMES = {
    "z": "位势",
    "q": "比湿",
    "t": "温度",
    "u": "纬向风",
    "v": "经向风",
}
SURFACE_NAMES = {
    "u10": "10 米纬向风",
    "v10": "10 米经向风",
    "t2m": "2 米温度",
    "mslp": "平均海平面气压",
}
UNIT_NAMES = {
    "m^2 s^-2": "m² s⁻²",
    "kg kg^-1": "kg kg⁻¹",
    "m s^-1": "m s⁻¹",
    "K": "K",
    "Pa": "Pa",
}


def _variable_name(variable: str) -> str:
    if variable in SURFACE_NAMES:
        return SURFACE_NAMES[variable]
    prefix = variable[0]
    if prefix not in PREFIX_NAMES or not variable[1:].isdigit():
        raise ValueError(f"unsupported variable name: {variable}")
    return f"{PREFIX_NAMES[prefix]}（{variable[1:]} hPa）"


def _format_rmse(value: float, unit: str) -> str:
    if unit == "kg kg^-1":
        return f"{value:.6e}"
    if abs(value) >= 100.0:
        return f"{value:.3f}"
    return f"{value:.6f}"


def _format_acc(value: float) -> str:
    return f"{value:.6f}"


def _bold_if(value: str, condition: bool) -> str:
    return f"**{value}**" if condition else value


def _load_rows(sources: list[Path]) -> tuple[list[str], dict[str, str], dict[tuple[str, str, str, int], tuple[float, float]]]:
    variables: list[str] = []
    units: dict[str, str] = {}
    values: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    row_count = 0
    for source in sources:
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row_count += 1
                if row["period"] != "2018-2019":
                    raise ValueError(f"unexpected period: {row['period']}")
                model = row["model"]
                method = row["method"]
                variable = row["variable"]
                lead = int(row["lead_time_hours"])
                if (
                    model not in MODEL_ORDER
                    or method not in METHODS_BY_MODEL[model]
                    or lead not in LEADS
                ):
                    raise ValueError(
                        f"unexpected result coordinate: {model}/{method}/{variable}/{lead}"
                    )
                if variable not in units:
                    variables.append(variable)
                    units[variable] = row["unit"]
                elif units[variable] != row["unit"]:
                    raise ValueError(f"unit mismatch for {variable}")
                key = (model, method, variable, lead)
                if key in values:
                    raise ValueError(f"duplicate result row: {key}")
                values[key] = (float(row["rmse"]), float(row["acc"]))

    expected = sum(len(methods) for methods in METHODS_BY_MODEL.values()) * len(variables) * len(LEADS)
    if len(variables) != 49 or row_count != expected or len(values) != expected:
        raise ValueError(
            f"incomplete source: variables={len(variables)}, rows={row_count}, "
            f"unique={len(values)}, expected={expected}"
        )
    for model, methods in METHODS_BY_MODEL.items():
        for method in methods:
            for variable in variables:
                for lead in LEADS:
                    if (model, method, variable, lead) not in values:
                        raise ValueError(
                            f"missing result row: {model}/{method}/{variable}/{lead}"
                        )
    return variables, units, values


def _render(
    sources: list[Path],
    variables: list[str],
    units: dict[str, str],
    values: dict[tuple[str, str, str, int], tuple[float, float]],
) -> str:
    lines = [
        "# 统一 11-Baseline、SphereDyn + SphereTTC：49 变量完整主实验表",
        "",
        "> 数据状态：冻结历史主实验结果  ",
        "> 评估时期：2018-01-01 至 2019-12-31  ",
        "> Initialization：730  ",
        "> 模型：原 11 个 baseline + SphereDyn-v9 seed 44  ",
        "> 方法：Raw、publication SphereTTC、SphereTTC-v22  ",
        "> 变量：49  ",
        "> 时效：24、72、120、168、240h",
        "",
        "## 阅读说明",
        "",
        "本文档联合冻结的原 11-baseline 主结果与 SphereDyn-v9 seed 44 主结果生成，共覆盖 `5,390 + 490 = 5,880` 条模型—方法—变量—时效结果。每个变量单独成表，避免对不同物理单位的 RMSE 做无意义的直接混合。",
        "",
        "每张表有 24 行：原 11 个 baseline 各有 Raw 与 `+ publication SphereTTC` 两行，表末另有 `SphereDyn-v9` 与 `SphereDyn-v9 + SphereTTC-v22` 两行。RMSE 越低越好，ACC 越高越好。SphereTTC 行中相对同模型 Raw 改善的数值使用粗体；未加粗不代表缺失，而是该指标在该时效没有改善。",
        "",
        "RMSE 是余弦纬度加权、在 730 个 initialization 上汇总后的物理量误差；ACC 使用仅由 1979–2016 fit split 构建的逐月日 climatology。publication SphereTTC 参数只在 2017 上选择；SphereTTC-v22 的 reference-anchor 权重也只使用 2017 拟合，二者均冻结后评估 2018–2019。",
        "",
        "生成源：" + "、".join(f"`{source.as_posix()}`" for source in sources),
        "",
        "## 变量索引",
        "",
    ]

    groups = (
        ("位势 z", [name for name in variables if name.startswith("z")]),
        ("比湿 q", [name for name in variables if name.startswith("q")]),
        ("温度 t", [name for name in variables if name.startswith("t") and name != "t2m"]),
        ("纬向风 u", [name for name in variables if name.startswith("u") and name != "u10"]),
        ("经向风 v", [name for name in variables if name.startswith("v") and name != "v10"]),
        ("地面变量", [name for name in variables if name in SURFACE_NAMES]),
    )
    for group_name, names in groups:
        links = "、".join(f"[{name}](#var-{name})" for name in names)
        lines.append(f"- {group_name}：{links}")
    lines.append("")

    header = ["模型 / 方法"]
    alignment = ["---"]
    for lead in LEADS:
        header.extend([f"{lead}h RMSE", f"{lead}h ACC"])
        alignment.extend(["---:", "---:"])

    for index, variable in enumerate(variables, start=1):
        unit = units[variable]
        display_unit = UNIT_NAMES.get(unit, unit)
        lines.extend(
            [
                f'<a id="var-{variable}"></a>',
                "",
                f"## {index}. `{variable}` — {_variable_name(variable)}",
                "",
                f"RMSE 单位：{display_unit}；ACC：无量纲。",
                "",
                "| " + " | ".join(header) + " |",
                "|" + "|".join(alignment) + "|",
            ]
        )
        for model in MODEL_ORDER:
            raw_cells = []
            ttc_cells = []
            for lead in LEADS:
                raw_rmse, raw_acc = values[(model, "raw", variable, lead)]
                final_method = METHODS_BY_MODEL[model][1]
                ttc_rmse, ttc_acc = values[(model, final_method, variable, lead)]
                raw_cells.extend(
                    [_format_rmse(raw_rmse, unit), _format_acc(raw_acc)]
                )
                ttc_cells.extend(
                    [
                        _bold_if(
                            _format_rmse(ttc_rmse, unit),
                            ttc_rmse < raw_rmse,
                        ),
                        _bold_if(
                            _format_acc(ttc_acc),
                            ttc_acc > raw_acc,
                        ),
                    ]
                )
            name = MODEL_NAMES[model]
            lines.append("| " + " | ".join([name, *raw_cells]) + " |")
            ttc_name = (
                f"{name} + SphereTTC-v22"
                if model == "spheredyn_v9"
                else f"{name} + SphereTTC"
            )
            lines.append(
                "| " + " | ".join([ttc_name, *ttc_cells]) + " |"
            )
        lines.extend(["", "[返回变量索引](#变量索引)", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=repository
        / "artifacts/ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "artifacts/spheredyn_spherettc_open_goal_20260801/MAIN_RESULTS_49_VARIABLES_ZH.md",
    )
    parser.add_argument(
        "--spheredyn-source",
        type=Path,
        default=repository
        / (
            "artifacts/spheredyn_spherettc_open_goal_20260801/"
            "main_prediction_v2_seed44/paper_metrics/"
            "SPHEREDYN_MAIN_RESULTS_RMSE_ACC.csv"
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    spheredyn_source = args.spheredyn_source.resolve()
    output = args.output.resolve()
    sources = [source, spheredyn_source]
    variables, units, values = _load_rows(sources)
    sources_for_report = [
        current.relative_to(repository)
        if current.is_relative_to(repository)
        else current
        for current in sources
    ]
    rendered = _render(sources_for_report, variables, units, values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        f"WROTE {output} variables={len(variables)} "
        f"table_rows={len(variables) * sum(len(methods) for methods in METHODS_BY_MODEL.values())}"
    )


if __name__ == "__main__":
    main()
