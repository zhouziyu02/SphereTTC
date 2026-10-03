#!/usr/bin/env python3
"""Derive the paper-facing summary files in results/ from the frozen artifacts.

The two headline (SOTA) results of the SphereCast project are:

  A. SphereTTC (final version, `src/spherettc.py`, `run_ttc.py --method sphere_ttc`)
     as a plug-in on 11 frozen forecasting backbones, 2018-2019;
  B. SphereDyn-v9 (seed 44) + SphereTTC-v22 (multi-provider constrained mode), 2018-2019.

Read-only with respect to artifacts/; standard library only; deterministic output.
verify_migration.py regenerates everything in memory and requires byte identity.

    python tools/build_results_summary.py            # (re)write results/
    python tools/build_results_summary.py --check    # compare only, write nothing
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
LATEST = ART / "graphcast_spherettc_20260804/LATEST_RESULTS.csv"
ORIGINAL_BASELINE = ART / "ttc_publication_corrected_20260726/results/MAIN_RESULTS.csv"
FROZEN_PARAMS = ART / "ttc_publication_corrected_20260726/FROZEN_PARAMETERS.json"
STATS = ART / "spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/frozen_inputs/s2s_daily_54var_stats.json"
GATE = ART / "spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/MAIN_GOAL_GATE.json"
SCHEDULE = ART / "graphcast_spherettc_20260804/schedule"
COMPARISON = ART / "graphcast_spherettc_20260804/COMPARISON_WITH_FROZEN_49_TABLES.json"
OUT = ROOT / "results"

LEADS = ["24", "72", "120", "168", "240"]
BACKBONES = ["convlstm", "transformer", "fno", "vit", "cirt", "climode",
             "fourcastnetv2", "oneforecast", "fuxi", "pangu", "graphcast"]
LOCAL = BACKBONES[:6]
SPHEREDYN = ("spheredyn_v9", "raw")
SPHEREDYN_TTC = ("spheredyn_v9", "sphere_ttc_v22")
HEADLINE = [("z500", "Z500"), ("t850", "T850"), ("t2m", "T2M"), ("u10", "U10")]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _group(rows):
    groups: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["method"]), {})[(row["lead_time_hours"], row["variable"])] = row
    return groups


STD = {n: float(v) for n, v in zip(*(json.loads(STATS.read_text(encoding="utf-8"))[k] for k in ("variables", "std")), strict=True)}
LATEST_G = _group(_read(LATEST))
ORIGINAL_G = _group(_read(ORIGINAL_BASELINE))


def nrmse(cells, lead=None) -> float:
    return statistics.fmean(float(r["rmse"]) / STD[k[1]] for k, r in cells.items() if lead in (None, k[0]))


def acc(cells, lead=None) -> float:
    return statistics.fmean(float(r["acc"]) for k, r in cells.items() if lead in (None, k[0]))


def _f(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _display(key) -> str:
    return next(iter(LATEST_G[key].values()))["model_display_name"]


def _counts(raw, cal) -> tuple[int, int, int]:
    better = sum(float(cal[k]["rmse"]) < float(raw[k]["rmse"]) for k in raw)
    worse = sum(float(cal[k]["rmse"]) > float(raw[k]["rmse"]) for k in raw)
    acc_better = sum(float(cal[k]["acc"]) > float(raw[k]["acc"]) for k in raw)
    return better, worse, acc_better


def _write_csv(fields, rows) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


# --------------------------------------------------------------------------- A
def spherettc_rows() -> list[dict[str, object]]:
    params = json.loads(FROZEN_PARAMS.read_text(encoding="utf-8"))["models"]
    l64 = json.loads((SCHEDULE / "FROZEN_PRIMARY_GLOBAL_2017.json").read_text(encoding="utf-8"))["selected_params"]
    rows = []
    for model in BACKBONES:
        raw, cal = LATEST_G[(model, "raw")], LATEST_G[(model, "sphere_ttc")]
        better, worse, acc_better = _counts(raw, cal)
        p = l64 if model == "graphcast" else params[model]["best_params"]
        rows.append({
            "backbone": model, "display_name": _display((model, "raw")),
            "group": "local" if model in LOCAL else "official",
            "raw_macro_nrmse": nrmse(raw), "spherettc_macro_nrmse": nrmse(cal),
            "macro_nrmse_gain_percent": 100.0 * (1.0 - nrmse(cal) / nrmse(raw)),
            "rmse_improved_cells": better, "rmse_worse_cells": worse,
            "raw_macro_acc": acc(raw), "spherettc_macro_acc": acc(cal),
            "acc_improved_cells": acc_better,
            "params": f"lmax={p['lmax']} memory={p['memory_size']} half_life={p['half_life']:g} strength={p['strength']:g}",
            "params_source": ("artifacts/graphcast_spherettc_20260804/schedule/configs/generated/l64_s08.json"
                              if model == "graphcast" else f"artifacts/ttc_publication_corrected_20260726/params/{model}.json"),
            **{f"gain_{lead}h_percent": 100.0 * (1.0 - nrmse(cal, lead) / nrmse(raw, lead)) for lead in LEADS},
        })
    return rows


def spherettc_csv(rows) -> str:
    out = []
    for r in rows:
        out.append({k: (_f(v, 6) if isinstance(v, float) and "percent" not in k else _f(v, 4) if isinstance(v, float) else v)
                    for k, v in r.items()})
    return _write_csv(list(rows[0]), out)


def spherettc_md(rows) -> str:
    gains = [r["macro_nrmse_gain_percent"] for r in rows]
    graphcast_original = 100.0 * (1.0 - nrmse(ORIGINAL_G[("graphcast", "sphere_ttc")]) / nrmse(LATEST_G[("graphcast", "raw")]))
    lines = [
        "# SphereTTC on 11 frozen backbones (2018–2019, 730 initializations)",
        "",
        "Final SphereTTC (`src/spherettc.py`, `scripts/run_ttc.py --method sphere_ttc`); per-backbone parameters",
        "selected on 2017 only and frozen. macro nRMSE = equal-weight mean over 49 variables × 5 leads of",
        "RMSE / (1979–2016 variable std). Generated by `tools/build_results_summary.py`.",
        "",
        "| Backbone | Raw nRMSE | +SphereTTC | Gain | RMSE better / worse cells | macro ACC Raw → +SphereTTC | Params |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['display_name']} | {r['raw_macro_nrmse']:.4f} | {r['spherettc_macro_nrmse']:.4f} | "
            f"**{r['macro_nrmse_gain_percent']:+.2f}%** | {r['rmse_improved_cells']} / {r['rmse_worse_cells']} | "
            f"{r['raw_macro_acc']:.4f} → {r['spherettc_macro_acc']:.4f} | {r['params']} |")
    lines += [
        "",
        f"- Backbones whose macro nRMSE improves: **{sum(g > 0 for g in gains)}/11**; "
        f"mean gain **{statistics.fmean(gains):.2f}%**, median {statistics.median(gains):.2f}%, "
        f"range {min(gains):.2f}% – {max(gains):.2f}%.",
        "- Mean gain by lead (11 backbones): " + ", ".join(
            f"{lead}h {statistics.fmean(r[f'gain_{lead}h_percent'] for r in rows):+.2f}% "
            f"({sum(r[f'gain_{lead}h_percent'] > 0 for r in rows)}/11 improved)" for lead in LEADS) + ".",
        "- macro ACC increases on " + str(sum(r['spherettc_macro_acc'] > r['raw_macro_acc'] for r in rows))
        + "/11 backbones; the exceptions are " + ", ".join(
            f"{r['display_name']} ({r['spherettc_macro_acc'] - r['raw_macro_acc']:+.5f})"
            for r in rows if r['spherettc_macro_acc'] <= r['raw_macro_acc']) + ".",
        f"- GraphCast uses the later 2017-selected global setting `l64_s08` (frozen 2026-08-04); the original "
        f"publication setting (lmax 24) gave {graphcast_original:+.2f}%. On an independent 2020 prospective "
        "holdout `l64_s08` has 0/245 RMSE-regressed cells and a mean cell RMSE gain of 3.57%.",
        "- FourCastNetV2 / FuXi gains are concentrated in specific humidity derived from RH (see docs/04 R5).",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- B
def spheredyn_rows() -> list[dict[str, object]]:
    rows = []
    keys = [k for k in LATEST_G]
    for lead in LEADS + [None]:
        raw, cal = LATEST_G[SPHEREDYN], LATEST_G[SPHEREDYN_TTC]
        raw_models = {m: nrmse(LATEST_G[(m, "raw")], lead) for m in BACKBONES}
        best_raw = min(raw_models, key=raw_models.get)
        best_local = min(LOCAL, key=raw_models.get)
        rank_raw = sorted(list(raw_models.values()) + [nrmse(raw, lead)]).index(nrmse(raw, lead)) + 1
        all_nrmse = sorted(nrmse(LATEST_G[k], lead) for k in keys)
        all_acc = sorted((acc(LATEST_G[k], lead) for k in keys), reverse=True)
        rows.append({
            "lead": "all" if lead is None else f"{lead}h",
            "spheredyn_raw_nrmse": nrmse(raw, lead), "spheredyn_spherettc_nrmse": nrmse(cal, lead),
            "gain_percent": 100.0 * (1.0 - nrmse(cal, lead) / nrmse(raw, lead)),
            "spheredyn_raw_acc": acc(raw, lead), "spheredyn_spherettc_acc": acc(cal, lead),
            "best_local_baseline": _display((best_local, "raw")), "best_local_raw_nrmse": raw_models[best_local],
            "best_raw_backbone": _display((best_raw, "raw")), "best_raw_nrmse": raw_models[best_raw],
            "graphcast_raw_nrmse": raw_models["graphcast"],
            "spheredyn_raw_rank_among_12_raw": rank_raw,
            "spheredyn_spherettc_nrmse_rank_among_24": all_nrmse.index(nrmse(cal, lead)) + 1,
            "spheredyn_spherettc_acc_rank_among_24": all_acc.index(acc(cal, lead)) + 1,
        })
    return rows


def spheredyn_csv(rows) -> str:
    return _write_csv(list(rows[0]), [
        {k: (_f(v, 6) if isinstance(v, float) and k != "gain_percent" else _f(v, 4) if isinstance(v, float) else v)
         for k, v in r.items()} for r in rows])


def _beats_local_everywhere() -> bool:
    sd = LATEST_G[SPHEREDYN]
    return all(
        nrmse(sd, lead) < nrmse(LATEST_G[(m, "raw")], lead) and acc(sd, lead) > acc(LATEST_G[(m, "raw")], lead)
        for m in LOCAL for lead in LEADS
    )


def _raw_acc_rank(lead: str) -> int:
    values = sorted((acc(LATEST_G[(m, "raw")], lead) for m in BACKBONES + ["spheredyn_v9"]), reverse=True)
    return values.index(acc(LATEST_G[SPHEREDYN], lead)) + 1


def spheredyn_md(rows) -> str:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    raw, cal = LATEST_G[SPHEREDYN], LATEST_G[SPHEREDYN_TTC]
    better, worse, acc_better = _counts(raw, cal)
    total = rows[-1]
    lines = [
        "# SphereDyn and SphereDyn + SphereTTC (2018–2019, 730 initializations, seed 44)",
        "",
        "SphereDyn-v9 multiscale checkpoint (`spheredyn_v9_multiscale.pt`); SphereTTC-v22 = multi-provider",
        "constrained combination mode (SphereDyn weight ≥ 0.5, five released forecasts as references, weights",
        "selected on 2017 and frozen). Ranks are over the 24 Raw/calibrated rows of the latest 49-variable table.",
        "Generated by `tools/build_results_summary.py`.",
        "",
        "| Lead | SphereDyn Raw | SphereDyn + SphereTTC | Gain | ACC Raw → +TTC | Best local baseline | GraphCast Raw | +TTC nRMSE rank | +TTC ACC rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lead']} | {r['spheredyn_raw_nrmse']:.4f} | **{r['spheredyn_spherettc_nrmse']:.4f}** | "
            f"{r['gain_percent']:+.2f}% | {r['spheredyn_raw_acc']:.4f} → {r['spheredyn_spherettc_acc']:.4f} | "
            f"{r['best_local_raw_nrmse']:.4f} ({r['best_local_baseline']}) | {r['graphcast_raw_nrmse']:.4f} | "
            f"{r['spheredyn_spherettc_nrmse_rank_among_24']}/24 | {r['spheredyn_spherettc_acc_rank_among_24']}/24 |")
    lines += [
        "",
        f"- Frozen main gate (`MAIN_GOAL_GATE.json`): macro nRMSE {gate['raw_spheredyn_macro_nrmse']:.7f} → "
        f"{gate['final_spheredyn_plus_spherettc_macro_nrmse']:.7f}, gain {gate['gain_percent']:.4f}% "
        f"(threshold {gate['minimum_gain_percent']:g}%), status `{gate['status']}`.",
        f"- SphereTTC improves RMSE in {better}/245 cells (worse in {worse}) and ACC in {acc_better}/245 cells.",
        f"- SphereDyn + SphereTTC has the **highest macro ACC of all 24 rows** ({total['spheredyn_spherettc_acc']:.4f}) and "
        f"the #{total['spheredyn_spherettc_nrmse_rank_among_24']} macro nRMSE; it is #1 on both metrics at 168h and 240h.",
        f"- Raw SphereDyn ({total['spheredyn_raw_nrmse']:.4f}) beats every locally trained baseline "
        f"(best: {total['best_local_baseline']} {total['best_local_raw_nrmse']:.4f}); "
        + ("it does so on both nRMSE and ACC at every lead. " if _beats_local_everywhere() else "")
        + "It has the lowest raw nRMSE of all 12 backbones at 240h, but its 240h ACC ranks "
        + f"{_raw_acc_rank('240')}/12, so this long-lead RMSE advantage partly reflects smoother forecasts.",
        "- Caveats: single seed; 2018–2019 was also used as development data for SphereDyn; the gain includes the",
        "  reference-forecast combination (see docs/04_AUDIT_AND_RISKS.md R2/R3).",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- shared tables
def macro_table() -> str:
    entries = []

    def add(model, method, cells, raw_cells, source, note):
        better, _, acc_better = _counts(raw_cells, cells) if method != "raw" else ("", "", "")
        gain = "" if method == "raw" else _f(100.0 * (1.0 - nrmse(cells) / nrmse(raw_cells)), 4)
        cell_gain = "" if method == "raw" else _f(100.0 * statistics.fmean(
            1.0 - float(cells[k]["rmse"]) / float(raw_cells[k]["rmse"]) for k in cells), 4)
        entries.append({
            "model": model, "model_display_name": next(iter(cells.values()))["model_display_name"],
            "method": method, "table": source, "macro_nrmse": nrmse(cells), "macro_acc": acc(cells),
            "macro_nrmse_gain_vs_raw_percent": gain, "mean_cell_rmse_gain_percent": cell_gain,
            "rmse_improved_cells_of_245": better, "acc_improved_cells_of_245": acc_better,
            "uses_external_reference_forecasts": "yes" if (model, method) == SPHEREDYN_TTC else "no",
            "note": note,
        })

    for (model, method), cells in LATEST_G.items():
        note = ""
        if (model, method) == SPHEREDYN_TTC:
            note = "SOTA B: SphereDyn + SphereTTC-v22 (multi-provider constrained mode); best macro ACC"
        elif method == "sphere_ttc":
            note = "SOTA A: SphereTTC plug-in" + (" (global l64_s08)" if model == "graphcast" else " (publication params)")
        elif (model, method) == SPHEREDYN:
            note = "SphereDyn backbone"
        add(model, method, cells, LATEST_G[(model, "raw")], "latest", note)
    add("graphcast", "sphere_ttc", ORIGINAL_G[("graphcast", "sphere_ttc")], ORIGINAL_G[("graphcast", "raw")],
        "frozen_original_superseded", "publication SphereTTC lmax=24 (superseded by l64_s08)")
    entries.sort(key=lambda e: (e["macro_nrmse"], e["table"]))
    fields = ["rank", "model", "model_display_name", "method", "table", "macro_nrmse", "macro_acc",
              "macro_nrmse_gain_vs_raw_percent", "mean_cell_rmse_gain_percent", "rmse_improved_cells_of_245",
              "acc_improved_cells_of_245", "uses_external_reference_forecasts", "note"]
    return _write_csv(fields, [dict(e, rank=i, macro_nrmse=_f(e["macro_nrmse"]), macro_acc=_f(e["macro_acc"]))
                               for i, e in enumerate(entries, 1)])


def headline(lead: str) -> str:
    lines = [
        f"# {lead}h headline table (latest, 2018–2019)",
        "",
        f"Raw / calibrated physical RMSE at {lead}h (lower is better). Z500 in m² s⁻², T850/T2M in K, U10 in m s⁻¹.",
        "Generated by `tools/build_results_summary.py` from `artifacts/graphcast_spherettc_20260804/LATEST_RESULTS.csv`.",
        "Calibration = SphereTTC (GraphCast: global `l64_s08`; others: publication parameters);",
        "† SphereDyn row = SphereDyn + SphereTTC-v22 (multi-provider constrained mode).",
        "",
        "| Model | Method | Z500 | T850 | T2M | U10 | Mean improvement |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in BACKBONES + ["spheredyn_v9"]:
        method = SPHEREDYN_TTC[1] if model == "spheredyn_v9" else "sphere_ttc"
        raw, cal = LATEST_G[(model, "raw")], LATEST_G[(model, method)]
        cells, gains = [], []
        for var, _ in HEADLINE:
            a, b = float(raw[(lead, var)]["rmse"]), float(cal[(lead, var)]["rmse"])
            digits = 2 if var == "z500" else 3
            cells.append(f"{a:.{digits}f} / {b:.{digits}f}")
            gains.append(1.0 - b / a)
        name = _display((model, "raw")) + (" †" if model == "spheredyn_v9" else "")
        lines.append(f"| {name} | {method} | " + " | ".join(cells) + f" | {100.0 * statistics.fmean(gains):+.2f}% |")
    return "\n".join(lines) + "\n"


def _cells_macro(path: Path) -> tuple[float, float]:
    rows = _read(path)
    calibrated = next(c for c in rows[0] if c.endswith("_rmse") and c != "raw_rmse")
    return (statistics.fmean(float(r["raw_rmse"]) / STD[r["variable"]] for r in rows),
            statistics.fmean(float(r[calibrated]) / STD[r["variable"]] for r in rows))


def sota_summary(a_rows, b_rows) -> str:
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    gate = load(GATE)
    frozen = load(SCHEDULE / "FROZEN_PRIMARY_GLOBAL_2017.json")
    prospective = load(SCHEDULE / "PROSPECTIVE_2020_SUMMARY.json")
    bootstrap = load(SCHEDULE / "PROSPECTIVE_2020_BOOTSTRAP.json")
    raw2020, cal2020 = _cells_macro(SCHEDULE / "PROSPECTIVE_2020_PRIMARY_CELLS.csv")
    gains = [r["macro_nrmse_gain_percent"] for r in a_rows]
    total = b_rows[-1]
    raw, cal = LATEST_G[SPHEREDYN], LATEST_G[SPHEREDYN_TTC]
    better, worse, acc_better = _counts(raw, cal)
    summary = {
        "evaluation": {"period": "2018-01-01/2019-12-31", "initializations": 730, "variables": 49,
                       "lead_hours": [24, 72, 120, 168, 240],
                       "primary_metric": "macro nRMSE: equal-weight mean of RMSE / 1979-2016 std over 49 variables x 5 leads",
                       "parameter_selection": "2017 only, frozen before 2018-2019"},
        "A_spherettc_final_version": {
            "code": "src/spherettc.py::SphereTTCCalibrator via scripts/run_ttc.py --method sphere_ttc",
            "backbones": len(a_rows),
            "backbones_improved_macro_nrmse": sum(g > 0 for g in gains),
            "mean_macro_nrmse_gain_percent": round(statistics.fmean(gains), 6),
            "median_macro_nrmse_gain_percent": round(statistics.median(gains), 6),
            "mean_gain_by_lead_percent": {lead: round(statistics.fmean(r[f"gain_{lead}h_percent"] for r in a_rows), 6) for lead in LEADS},
            "backbones_improved_macro_acc": sum(r["spherettc_macro_acc"] > r["raw_macro_acc"] for r in a_rows),
            "per_backbone": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
                              if not k.startswith("gain_") and k not in ("display_name",)} for r in a_rows],
            "graphcast_l64_s08": {
                "params": frozen["selected_params"], "frozen_at": frozen["created_at"],
                "prospective_2020": {"macro_nrmse_raw": round(raw2020, 10), "macro_nrmse_spherettc": round(cal2020, 10),
                                     "mean_cell_rmse_gain": prospective["primary_global"]["mean_relative_rmse_gain"],
                                     "rmse_regressed_cells": prospective["primary_global"]["negative_rmse_cells"],
                                     "mean_acc_delta": prospective["primary_global"]["mean_acc_delta"],
                                     "bootstrap_95ci_mean_cell_rmse_gain": bootstrap["primary_global"]["mean_relative_rmse_gain_percentiles_2p5_50_97p5"][0::2]},
            },
            "table": "results/SPHERETTC_11_BACKBONES.md",
        },
        "B_spheredyn_plus_spherettc": {
            "backbone_checkpoint": "artifacts/spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt",
            "calibration_weights": "artifacts/spheredyn_spherettc_open_goal_20260801/main_prediction_v2_seed44/final/FITTED_WEIGHTS_2017.npz",
            "code": "backbone src/spheredyn.py::SphereDyn; calibration src/spherettc.py Part 4 (Mode B), run by scripts/spheredyn/run_main_spherettc_v2.py; launcher scripts/spheredyn/run_main_prediction_v2_4gpu.sh",
            "spherettc_mode": "SphereTTC-v22: multi-provider constrained combination (SphereDyn weight >= 0.5; references FourCastNetV2, OneForecast, FuXi, Pangu, GraphCast; fitted on 2017, frozen)",
            "raw_macro_nrmse": gate["raw_spheredyn_macro_nrmse"],
            "spheredyn_plus_spherettc_macro_nrmse": gate["final_spheredyn_plus_spherettc_macro_nrmse"],
            "gain_percent": gate["gain_percent"],
            "gate_status": gate["status"],
            "raw_macro_acc": round(acc(raw), 6), "spheredyn_plus_spherettc_macro_acc": round(acc(cal), 6),
            "rmse_improved_cells": better, "rmse_worse_cells": worse, "acc_improved_cells": acc_better,
            "macro_nrmse_rank_among_24": total["spheredyn_spherettc_nrmse_rank_among_24"],
            "macro_acc_rank_among_24": total["spheredyn_spherettc_acc_rank_among_24"],
            "by_lead": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in b_rows[:-1]],
            "table": "results/SPHEREDYN_SPHERETTC.md",
        },
        "open_risks": "docs/04_AUDIT_AND_RISKS.md (R1 daily-mean memory admission must be fixed and re-run before submission)",
    }
    return json.dumps(summary, indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- paper-style tables
def _paper_rows():
    return [(m, "sphere_ttc") for m in BACKBONES] + [SPHEREDYN_TTC]


def paper_table_nrmse() -> str:
    lines = [
        "# Table 1 (draft) · macro nRMSE by lead: Raw → +SphereTTC (relative change)",
        "",
        "2018–2019, 730 initializations, 49 variables; lower is better. Calibration parameters selected on 2017 only.",
        "Last row = SphereDyn (ours) + SphereTTC (multi-provider constrained mode). Generated by `tools/build_results_summary.py`.",
        "",
        "| Backbone | 24h | 72h | 120h | 168h | 240h | All |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in _paper_rows():
        raw, cal = LATEST_G[(model, "raw")], LATEST_G[(model, method)]
        cells = []
        for lead in LEADS + [None]:
            a, b = nrmse(raw, lead), nrmse(cal, lead)
            cells.append(f"{a:.4f} → {b:.4f} ({100.0 * (b / a - 1.0):+.2f}%)")
        name = _display((model, "raw")) + (" (ours)" if model == "spheredyn_v9" else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paper_table_acc() -> str:
    lines = [
        "# Table 2 (draft) · macro ACC by lead: Raw → +SphereTTC (absolute change)",
        "",
        "2018–2019, 730 initializations, 49 variables; higher is better; climatology from 1979–2016 only.",
        "Generated by `tools/build_results_summary.py`.",
        "",
        "| Backbone | 24h | 72h | 120h | 168h | 240h | All |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, method in _paper_rows():
        raw, cal = LATEST_G[(model, "raw")], LATEST_G[(model, method)]
        cells = []
        for lead in LEADS + [None]:
            a, b = acc(raw, lead), acc(cal, lead)
            cells.append(f"{a:.4f} → {b:.4f} ({b - a:+.4f})")
        name = _display((model, "raw")) + (" (ours)" if model == "spheredyn_v9" else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def efficiency_md() -> str:
    from datetime import datetime

    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    prof = SCHEDULE / "profiles"
    raw20, ttc20 = load(prof / "holdout_2020_raw.json"), load(prof / "holdout_2020_l64_s08.json")
    raw18, ttc18 = load(prof / "retrospective_raw_2018.json"), load(prof / "retrospective_l64_s08_2018.json")
    gc = load(prof / "graphcast_2020_current_a100.json")
    gc_seconds = (datetime.fromisoformat(gc["finished_at"]) - datetime.fromisoformat(gc["started_at"])).total_seconds()
    n20, n18 = ttc20["pred_shape"][0], ttc18["pred_shape"][0]
    params = json.loads(FROZEN_PARAMS.read_text(encoding="utf-8"))["models"]
    selection = [params[m]["best_validation_result"]["wall_seconds"] for m in BACKBONES]
    design = load(ART / "spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/DESIGN.json")
    ckpt = ART / "spheredyn_spherettc_open_goal_20260801/spheredyn_v9_h100_paired_screen/spheredyn_v9_multiscale_seed44/checkpoints/spheredyn_v9_multiscale.pt"
    per_init_20 = ttc20["wall_seconds"] / n20
    extra_20 = (ttc20["wall_seconds"] - raw20["wall_seconds"]) / n20
    lines = [
        "# Efficiency (draft)",
        "",
        "All numbers are read from frozen run profiles; generated by `tools/build_results_summary.py`.",
        "SphereTTC timings are for GraphCast + SphereTTC `l64_s08` (49 variables × 5 leads × 121 × 240 grid, one A100-80GB),",
        "measured end-to-end including truth loading and RMSE/ACC scoring.",
        "",
        "| Item | Value | Source |",
        "|---|---:|---|",
        "| SphereTTC learnable parameters / gradient training | 0 / none (closed-form weighted ridge fit per step) | `src/spherettc.py` |",
        f"| SphereTTC end-to-end time per initialization, 2020 | {per_init_20:.3f} s | `holdout_2020_l64_s08.json` ({ttc20['wall_seconds']:.1f} s / {n20}) |",
        f"| … of which calibration overhead over raw scoring, 2020 | {extra_20:.3f} s | minus `holdout_2020_raw.json` ({raw20['wall_seconds']:.1f} s) |",
        f"| … same for 2018 | {ttc18['wall_seconds'] / n18:.3f} s ({(ttc18['wall_seconds'] - raw18['wall_seconds']) / n18:.3f} s overhead) | `retrospective_*_2018.json` |",
        f"| SphereTTC peak GPU memory | {ttc20['cuda_peak_memory_gb']:.2f} GB (raw scoring {raw20['cuda_peak_memory_gb']:.2f} GB) | same profiles |",
        f"| GraphCast-small 1° inference per initialization (10-day rollout, incl. ERA5 loading), 2020 | {gc_seconds / n20:.1f} s | `graphcast_2020_current_a100.json` ({gc_seconds:.0f} s / {n20}) |",
        f"| SphereTTC overhead relative to GraphCast inference | {100.0 * extra_20 / (gc_seconds / n20):.2f}% (end-to-end {100.0 * per_init_20 / (gc_seconds / n20):.2f}%) | ratio of the rows above |",
        f"| Hyper-parameter selection cost per candidate per backbone (2017, 245 scored inits, V100) | {min(selection):.0f}–{max(selection):.0f} s | `FROZEN_PARAMETERS.json` |",
        f"| SphereDyn-v9 parameters | {design['architecture']['parameter_count']:,} | `spheredyn_v9_h100_paired_screen/DESIGN.json` |",
        f"| SphereDyn-v9 checkpoint size | {ckpt.stat().st_size:,} bytes | `spheredyn_v9_multiscale.pt` |",
        f"| SphereDyn-v9 final training stage | {design['training']['fit_samples']} samples × {design['training']['epochs']} epochs, batch {design['training']['batch_size']}, lr {design['training']['learning_rate']}, warm-started from v8 | `DESIGN.json` |",
        "",
        "Not available in this package: SphereDyn total training GPU-hours, MACs/FLOPs of any model, inference time of the other 10 backbones.",
    ]
    return "\n".join(lines) + "\n"


def build() -> dict[str, str]:
    a_rows, b_rows = spherettc_rows(), spheredyn_rows()
    return {
        "SPHERETTC_11_BACKBONES.md": spherettc_md(a_rows),
        "SPHERETTC_11_BACKBONES.csv": spherettc_csv(a_rows),
        "SPHEREDYN_SPHERETTC.md": spheredyn_md(b_rows),
        "SPHEREDYN_SPHERETTC.csv": spheredyn_csv(b_rows),
        "MACRO_SUMMARY_2018_2019.csv": macro_table(),
        "HEADLINE_120H_LATEST.md": headline("120"),
        "HEADLINE_240H_LATEST.md": headline("240"),
        "SOTA_SUMMARY.json": sota_summary(a_rows, b_rows),
        "PAPER_TABLE1_NRMSE_BY_LEAD.md": paper_table_nrmse(),
        "PAPER_TABLE2_ACC_BY_LEAD.md": paper_table_acc(),
        "EFFICIENCY.md": efficiency_md(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build()
    if args.check:
        bad = [n for n, text in outputs.items()
               if not (OUT / n).is_file() or (OUT / n).read_text(encoding="utf-8") != text]
        if bad:
            sys.exit(f"results/ out of date: {bad}")
        print("RESULTS_SUMMARY_UP_TO_DATE")
        return
    OUT.mkdir(exist_ok=True)
    for name, text in outputs.items():
        (OUT / name).write_text(text, encoding="utf-8")
    print("RESULTS_SUMMARY_WRITTEN " + " ".join(sorted(outputs)))


if __name__ == "__main__":
    main()
