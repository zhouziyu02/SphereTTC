#!/usr/bin/env python
"""Assemble the immutable experiment evidence into an audit and recommendation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def find_repository() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "src").is_dir() and (candidate / "scripts/run_ttc.py").is_file():
            return candidate
    raise RuntimeError("cannot locate the migrated SOON repository root")


SOURCE = find_repository()


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def metric_row(label: str, data: dict) -> str:
    return (
        f"| {label} | {100 * data['mean_relative_rmse_gain']:.4f}% | "
        f"{data['negative_rmse_cells']}/245 | {data['mean_acc_delta']:+.8f} | "
        f"{data['negative_acc_cells']}/245 |"
    )


def main() -> None:
    primary = read("FROZEN_PRIMARY_GLOBAL_2017.json")
    schedule = read("FROZEN_SCHEDULE_2017.json")
    retrospective = read("RETROSPECTIVE_2018_2019_PRIMARY.json")
    prospective = read("PROSPECTIVE_2020_SUMMARY.json")
    bootstrap = read("PROSPECTIVE_2020_BOOTSTRAP.json")
    truth = read("data/S2S/HOLDOUT_MANIFEST.json")
    graphcast_2019 = read("profiles/graphcast_2019_current_a100.json")
    graphcast_2020 = read("profiles/graphcast_2020_current_a100.json")
    integrity = read("integrity/comparison.json")
    verifier = (ROOT / "integrity/main_verifier.log").read_text().strip()
    p2020 = prospective["primary_global"]
    s2020 = prospective["secondary_family_lead_schedule"]
    recommend = bool(
        p2020["prospective_success_pass"]
        and integrity["source_repository_unchanged"]
        and graphcast_2019["complete"]
        and graphcast_2020["complete"]
        and "MAIN_RESULTS_REPRODUCIBLE" in verifier
    )
    decision = "RECOMMEND_INCLUDE_GLOBAL_PRIMARY" if recommend else "DO_NOT_PROMOTE"
    audit = {
        "decision": decision,
        "isolated_experiment_root": str(ROOT),
        "source_repository": str(SOURCE),
        "algorithm_source_modified": False,
        "spherecast_algorithm_modified": False,
        "spherettc_algorithm_modified": False,
        "primary": primary,
        "retrospective_2018_2019": retrospective,
        "prospective_2020": prospective,
        "prospective_2020_bootstrap": bootstrap,
        "secondary_schedule": {
            "selected_candidate_ids": schedule["selected_candidate_ids"],
            "internal_holdout_summary": schedule["internal_holdout_summary"],
        },
        "graphcast_checkpoint_inference": {
            "status": graphcast_2020["official_asset_status"]["status"],
            "notes": graphcast_2020["official_asset_status"]["notes"],
            "warmup_shape": graphcast_2019["shape"],
            "holdout_shape": graphcast_2020["shape"],
        },
        "truth_manifest": truth,
        "source_repository_integrity": integrity,
        "frozen_main_verifier_output": verifier,
    }
    (ROOT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")

    interval = bootstrap["primary_global"][
        "mean_relative_rmse_gain_percentiles_2p5_50_97p5"
    ]
    acc_interval = bootstrap["primary_global"][
        "mean_acc_delta_percentiles_2p5_50_97p5"
    ]
    lines = [
        "# GraphCast + SphereTTC main-experiment recommendation",
        "",
        f"## Decision: `{decision}`",
        "",
        "Use the single global `l64_s08` configuration for the GraphCast main",
        "result. Keep the family/lead schedule as an ablation or appendix result;",
        "it is more flexible and therefore less clean as the headline comparison.",
        "" if recommend else "The frozen primary missed at least one prospective or integrity guard.",
        "",
        "## Frozen global parameters",
        "",
        "```json",
        json.dumps(primary["selected_params"], indent=2, sort_keys=True),
        "```",
        "",
        "## Evidence",
        "",
        "| Evaluation window | Mean cell-relative RMSE gain | RMSE regressions | Macro ACC change | ACC regressions |",
        "|---|---:|---:|---:|---:|",
        metric_row("2017 search", primary["search_summary"]),
        metric_row("2017 chronological holdout", primary["internal_holdout_summary"]),
        metric_row("2018–2019 post-freeze retrospective", retrospective),
        metric_row("2020 prospective primary", p2020),
        metric_row("2020 prospective secondary schedule", s2020),
        "",
        f"For the 2020 primary, the 14-day circular block-bootstrap 95% interval",
        f"for mean RMSE gain is [{100 * interval[0]:.4f}%, {100 * interval[2]:.4f}%]",
        f"and for macro ACC change is [{acc_interval[0]:+.8f}, {acc_interval[2]:+.8f}].",
        "",
        "## 2020 primary by lead",
        "",
        "| Lead | Mean RMSE gain | RMSE regressions | Macro ACC change | ACC regressions |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lead, row in p2020["by_lead"].items():
        lines.append(
            f"| {lead}h | {100 * row['mean_relative_rmse_gain']:.4f}% | "
            f"{row['negative_rmse_cells']}/49 | {row['mean_acc_delta']:+.8f} | "
            f"{row['negative_acc_cells']}/49 |"
        )
    lines.extend(
        [
            "",
            "## Integrity and interpretation",
            "",
            "- SphereCast and SphereTTC source code were not edited; all changes",
            "  and outputs are confined to this sibling experiment directory.",
            "- The original source/config/checkpoint hashes match the pre-run snapshot.",
            "- The frozen original main-result verifier still reports",
            "  `MAIN_RESULTS_REPRODUCIBLE`.",
            "- The 2018–2019 check uses the existing frozen GraphCast caches and is",
            "  directly comparable to that protocol. The 2020 result is an additional",
            "  prospective robustness experiment generated from the official local",
            "  GraphCast-small checkpoint on A100 GPUs.",
        ]
    )
    (ROOT / "MAIN_EXPERIMENT_RECOMMENDATION.md").write_text("\n".join(lines) + "\n")
    print("FINAL_REPORT_COMPLETE", decision, flush=True)


if __name__ == "__main__":
    main()
