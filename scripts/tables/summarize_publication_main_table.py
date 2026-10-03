#!/usr/bin/env python
"""Create physical-unit RMSE and ACC tables for the frozen publication test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACC_KEYS = (
    "acc_numerator",
    "acc_prediction_energy",
    "acc_target_energy",
)
HEADLINE_VARIABLES = ("z500", "t850", "u850", "v850", "t2m", "mslp")
MANUSCRIPT_VARIABLES = ("z500", "t850", "t2m", "u10")
MODEL_DISPLAY_NAMES = {
    "climode": "ClimODE-style adapter",
    "graphcast": "GraphCast-small 1°",
}


def load_metrics(root: Path, model: str, method: str) -> dict[str, np.ndarray]:
    path = root / model / method / "per_initialization_metrics.npz"
    with np.load(path, allow_pickle=False) as source:
        missing = [name for name in ("init_time", "lead_time", "variable", "mse", *ACC_KEYS) if name not in source]
        if missing:
            raise RuntimeError(f"{path} is missing metrics required for RMSE/ACC: {missing}")
        return {name: source[name] for name in source.files}


def variable_unit(variable: str) -> str:
    if variable.startswith("z"):
        return "m^2 s^-2"
    if variable.startswith("q"):
        return "kg kg^-1"
    if variable.startswith("t"):
        return "K"
    if variable.startswith(("u", "v")):
        return "m s^-1"
    if variable == "mslp":
        return "Pa"
    raise ValueError(f"unit is not defined for variable {variable}")


def aggregate(data: dict[str, np.ndarray], keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(keep):
        raise RuntimeError("cannot aggregate an empty initialization selection")
    rmse = np.sqrt(data["mse"][keep].mean(axis=0))
    numerator = data["acc_numerator"][keep].sum(axis=0)
    prediction_energy = data["acc_prediction_energy"][keep].sum(axis=0)
    target_energy = data["acc_target_energy"][keep].sum(axis=0)
    denominator = np.sqrt(np.maximum(prediction_energy * target_energy, 1e-24))
    acc = numerator / denominator
    return rmse, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ttc-method", default="sphere_ttc")
    args = parser.parse_args()

    root = Path(args.metrics_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    aggregates: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    combined_period_label: str | None = None

    for model in args.models:
        raw = load_metrics(root, model, "raw")
        ttc = load_metrics(root, model, args.ttc_method)
        for key in ("init_time", "lead_time", "variable"):
            if not np.array_equal(raw[key], ttc[key]):
                raise RuntimeError(f"{model}: paired evaluation mismatch for {key}")

        years = raw["init_time"].astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
        unique_years = sorted(set(years.tolist()))
        model_period_label = (
            str(unique_years[0])
            if len(unique_years) == 1
            else f"{unique_years[0]}-{unique_years[-1]}"
        )
        if combined_period_label is None:
            combined_period_label = model_period_label
        elif combined_period_label != model_period_label:
            raise RuntimeError(
                "models do not share one combined evaluation period: "
                f"{combined_period_label} vs {model_period_label}"
            )
        periods = [
            (
                model_period_label,
                np.ones(raw["init_time"].size, dtype=bool),
                True,
            )
        ]
        if len(unique_years) > 1:
            periods.extend(
                (str(year), years == year, False)
                for year in unique_years
            )
        for method, data in (("raw", raw), (args.ttc_method, ttc)):
            for period, keep, is_combined in periods:
                rmse, acc = aggregate(data, keep)
                if is_combined:
                    aggregates[(model, method)] = (
                        rmse,
                        acc,
                        data["lead_time"],
                        data["variable"],
                    )
                rows = combined_rows if is_combined else period_rows
                for lead_index, lead in enumerate(data["lead_time"]):
                    for variable_index, variable_value in enumerate(data["variable"]):
                        variable = str(variable_value)
                        rows.append(
                            {
                                "model": model,
                                "model_display_name": MODEL_DISPLAY_NAMES.get(model, model),
                                "method": method,
                                "period": period,
                                "lead_time_hours": int(lead),
                                "variable": variable,
                                "unit": variable_unit(variable),
                                "rmse": float(rmse[lead_index, variable_index]),
                                "acc": float(acc[lead_index, variable_index]),
                                "n_initializations": int(keep.sum()),
                            }
                        )

    combined = pd.DataFrame(combined_rows)
    by_year = pd.DataFrame(period_rows)
    key_columns = ["model", "method", "period", "lead_time_hours", "variable"]
    duplicate_rows = int(combined.duplicated(key_columns).sum())
    nonfinite_values = int(
        (~np.isfinite(combined[["rmse", "acc"]].to_numpy(dtype=float))).sum()
    )
    if duplicate_rows or nonfinite_values:
        raise RuntimeError(
            f"invalid final table: duplicate_rows={duplicate_rows}, "
            f"nonfinite_values={nonfinite_values}"
        )
    combined.to_csv(output / "MAIN_RESULTS.csv", index=False)
    by_year.to_csv(output / "MAIN_RESULTS_BY_YEAR.csv", index=False)

    wide = combined.pivot(
        index=["model", "model_display_name", "method", "variable", "unit"],
        columns="lead_time_hours",
        values=["rmse", "acc"],
    )
    wide.columns = [f"{metric}_{lead}h" for metric, lead in wide.columns]
    wide.reset_index().to_csv(output / "MAIN_RESULTS_WIDE.csv", index=False)

    lines = [
        "# Main results: physical RMSE and ACC",
        "",
        "SphereTTC parameters were selected before evaluation on "
        f"{combined_period_label}.",
        "RMSE is cosine-latitude weighted and reported in each variable's native physical unit.",
        "ACC is cosine-latitude weighted against the 1979–2016 fit-split daily",
        "month/day climatology; no validation or test targets enter the climatology.",
        "RMSE is not averaged across variables with incompatible units; the complete 49-variable,",
        "5-lead table is in `MAIN_RESULTS.csv`, with year-specific values in",
        "`MAIN_RESULTS_BY_YEAR.csv`.",
        "",
    ]

    for variable in HEADLINE_VARIABLES:
        unit = variable_unit(variable)
        lines.extend(
            [
                "",
                f"## {variable} at 120h ({unit})",
                "",
                "| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model in args.models:
            raw_rmse, raw_acc, leads, variables = aggregates[(model, "raw")]
            ttc_rmse, ttc_acc, _, _ = aggregates[(model, args.ttc_method)]
            lead_index = int(np.where(leads == 120)[0][0])
            variable_index = int(np.where(variables == variable)[0][0])
            lines.append(
                f"| {MODEL_DISPLAY_NAMES.get(model, model)} | "
                f"{raw_rmse[lead_index, variable_index]:.8g} | "
                f"{ttc_rmse[lead_index, variable_index]:.8g} | "
                f"{raw_acc[lead_index, variable_index]:.6f} | "
                f"{ttc_acc[lead_index, variable_index]:.6f} |"
            )

    (output / "MAIN_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manuscript_rows = []
    all_lead_rows = []
    for model in args.models:
        raw_rmse, raw_acc, leads, variables = aggregates[(model, "raw")]
        ttc_rmse, ttc_acc, _, _ = aggregates[(model, args.ttc_method)]
        row = {
            "model": model,
            "model_display_name": MODEL_DISPLAY_NAMES.get(model, model),
            "lead_time_hours": 120,
        }
        relative_improvements = []
        for variable in MANUSCRIPT_VARIABLES:
            variable_index = int(np.where(variables == variable)[0][0])
            for lead_index, lead in enumerate(leads):
                raw_value = float(raw_rmse[lead_index, variable_index])
                ttc_value = float(ttc_rmse[lead_index, variable_index])
                all_lead_rows.append(
                    {
                        "model": model,
                        "model_display_name": MODEL_DISPLAY_NAMES.get(model, model),
                        "lead_time_hours": int(lead),
                        "variable": variable,
                        "unit": variable_unit(variable),
                        "raw_rmse": raw_value,
                        "sphere_ttc_rmse": ttc_value,
                        "rmse_change_percent": 100.0 * (ttc_value / raw_value - 1.0),
                        "raw_acc": float(raw_acc[lead_index, variable_index]),
                        "sphere_ttc_acc": float(ttc_acc[lead_index, variable_index]),
                        "acc_change": float(
                            ttc_acc[lead_index, variable_index]
                            - raw_acc[lead_index, variable_index]
                        ),
                    }
                )
            lead_index = int(np.where(leads == 120)[0][0])
            raw_value = float(raw_rmse[lead_index, variable_index])
            ttc_value = float(ttc_rmse[lead_index, variable_index])
            row[f"{variable}_raw_rmse"] = raw_value
            row[f"{variable}_sphere_ttc_rmse"] = ttc_value
            row[f"{variable}_raw_acc"] = float(raw_acc[lead_index, variable_index])
            row[f"{variable}_sphere_ttc_acc"] = float(
                ttc_acc[lead_index, variable_index]
            )
            relative_improvements.append(100.0 * (1.0 - ttc_value / raw_value))
        row["mean_rmse_improvement_percent"] = float(
            np.mean(relative_improvements)
        )
        manuscript_rows.append(row)

    manuscript = pd.DataFrame(manuscript_rows)
    manuscript.to_csv(output / "MANUSCRIPT_MAIN_TABLE_120H.csv", index=False)
    pd.DataFrame(all_lead_rows).to_csv(
        output / "MANUSCRIPT_ALL_LEADS_HEADLINE.csv", index=False
    )
    compact_lines = [
        "# Manuscript main table: day-5 RMSE",
        "",
        "Values are Raw / SphereTTC. Z500 is in m² s⁻², temperatures in K,",
        "and U10 in m s⁻¹. Lower is better.",
        "",
        "| Model | Z500 | T850 | T2M | U10 | Mean improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in manuscript_rows:
        compact_lines.append(
            f"| {row['model_display_name']} | "
            f"{row['z500_raw_rmse']:.2f} / {row['z500_sphere_ttc_rmse']:.2f} | "
            f"{row['t850_raw_rmse']:.3f} / {row['t850_sphere_ttc_rmse']:.3f} | "
            f"{row['t2m_raw_rmse']:.3f} / {row['t2m_sphere_ttc_rmse']:.3f} | "
            f"{row['u10_raw_rmse']:.3f} / {row['u10_sphere_ttc_rmse']:.3f} | "
            f"{row['mean_rmse_improvement_percent']:+.2f}% |"
        )
    (output / "MANUSCRIPT_MAIN_TABLE_120H.md").write_text(
        "\n".join(compact_lines) + "\n", encoding="utf-8"
    )

    audit = {
        "models": len(args.models),
        "methods": 2,
        "variables": int(combined["variable"].nunique()),
        "lead_times": sorted(
            combined["lead_time_hours"].unique().astype(int).tolist()
        ),
        "rows": int(len(combined)),
        "expected_rows": len(args.models) * 2 * 49 * 5,
        "duplicate_key_rows": duplicate_rows,
        "nonfinite_rmse_or_acc_values": nonfinite_values,
        "n_initializations": sorted(
            combined["n_initializations"].unique().astype(int).tolist()
        ),
    }
    audit["complete"] = bool(
        audit["variables"] == 49
        and audit["lead_times"] == [24, 72, 120, 168, 240]
        and audit["rows"] == audit["expected_rows"]
        and duplicate_rows == 0
        and nonfinite_values == 0
    )
    (output / "TABLE_COMPLETENESS_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {output / 'MAIN_RESULTS.csv'}")
    print(f"WROTE {output / 'MAIN_RESULTS_BY_YEAR.csv'}")
    print(f"WROTE {output / 'MAIN_RESULTS_WIDE.csv'}")
    print(f"WROTE {output / 'MAIN_RESULTS.md'}")
    print(f"WROTE {output / 'MANUSCRIPT_MAIN_TABLE_120H.csv'}")
    print(f"WROTE {output / 'MANUSCRIPT_MAIN_TABLE_120H.md'}")
    print(f"WROTE {output / 'MANUSCRIPT_ALL_LEADS_HEADLINE.csv'}")
    print(f"WROTE {output / 'TABLE_COMPLETENESS_AUDIT.json'}")


if __name__ == "__main__":
    main()
