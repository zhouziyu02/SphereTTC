#!/usr/bin/env python3
"""Build paired main-experiment tables from one results.json per forecaster."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from spherettc.evaluation import relative_improvements


def summarize(results: list[dict]) -> tuple[list[dict], list[dict]]:
    if not results:
        raise ValueError("At least one experiment result is required.")
    shared_keys = (
        "variables", "lead_time_hours", "start", "end", "n_initializations",
        "initialization_sha256", "verification_delay_hours", "statistics_sha256", "grid",
        "climatology_metadata", "source_kinds",
    )
    reference = results[0]
    names = set()
    rows, overall = [], []
    for result in results:
        if result["model"] in names:
            raise ValueError(f"Duplicate model result: {result['model']}")
        names.add(result["model"])
        for key in shared_keys:
            if result.get(key) != reference.get(key):
                raise ValueError(f"Cannot compare different protocols: {key} differs for {result['model']}.")
        raw, corrected = result["macro"]["raw"], result["macro"]["spherettc"]
        for index, lead in enumerate(result["lead_time_hours"]):
            row = {"model": result["model"], "lead_time_hours": lead}
            for method, values in (("raw", raw), ("spherettc", corrected)):
                row[f"{method}_nrmse"] = values["nrmse_by_lead"][index]
                row[f"{method}_acc"] = values["acc_by_lead"][index]
            row["nrmse_improvement_percent"], row["acc_improvement_percent"] = relative_improvements(
                row["raw_nrmse"], row["spherettc_nrmse"], row["raw_acc"], row["spherettc_acc"],
            )
            rows.append(row)
        row = {"model": result["model"]}
        for method, values in (("raw", raw), ("spherettc", corrected)):
            row[f"{method}_nrmse"] = values["overall_nrmse"]
            row[f"{method}_acc"] = values["overall_acc"]
        row["nrmse_improvement_percent"], row["acc_improvement_percent"] = relative_improvements(
            row["raw_nrmse"], row["spherettc_nrmse"], row["raw_acc"], row["spherettc_acc"],
        )
        overall.append(row)
    return rows, overall


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value, digits=4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, help="One results.json for each evaluated model.")
    parser.add_argument("--output", required=True, help="New directory for CSV and Markdown tables.")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}; select a new directory.")
    results = [json.loads(Path(path).read_text()) for path in args.results]
    rows, overall = summarize(results)
    output.mkdir(parents=True)
    write_csv(output / "main_results.csv", rows)
    write_csv(output / "overall_results.csv", overall)
    text = [
        "# Main experiment results", "",
        "nRMSE and ACC average variables equally; overall scores also average lead times equally.",
        "nRMSE improvement = 100 (1 - calibrated / raw); ACC improvement = 100 (calibrated / raw - 1).",
        "The relative change is undefined when its reference denominator is zero (or nonpositive for ACC).", "",
        "| Forecaster | Lead (days) | Raw nRMSE | +SphereTTC | Improvement (%) | Raw ACC | +SphereTTC | Improvement (%) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        text.append("| " + " | ".join([
            row["model"], f"{row['lead_time_hours'] / 24:g}",
            format_value(row["raw_nrmse"]), format_value(row["spherettc_nrmse"]),
            format_value(row["nrmse_improvement_percent"], 2), format_value(row["raw_acc"]),
            format_value(row["spherettc_acc"]), format_value(row["acc_improvement_percent"], 2),
        ]) + " |")
    gains = [row["nrmse_improvement_percent"] for row in overall if row["nrmse_improvement_percent"] is not None]
    text += ["", f"Mean overall relative nRMSE improvement across {len(gains)} equally weighted forecasters: {np.mean(gains):.4f}%."]
    (output / "main_results.md").write_text("\n".join(text) + "\n")
    print(f"Wrote tables for {len(results)} forecasters to {output}.")


if __name__ == "__main__":
    main()
