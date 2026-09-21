#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.validation import evaluate_mrr


def _blend_predictions(
    frame: pd.DataFrame,
    mass_weight: float,
    rrf_k: float,
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        values = group.copy()
        values["blend_score"] = (
            mass_weight / (rrf_k + values["mass_rank"].astype(float))
            + (1.0 - mass_weight) / (rrf_k + values["rank"].astype(float))
        )
        ordered = values.sort_values(
            ["blend_score", "mass_rank"], ascending=[False, True], kind="stable"
        )
        output[str(molecule_id)] = ordered["inchikey14"].astype(str).tolist()
    return output


def _truth_from_manifest(prediction_path: Path, frame: pd.DataFrame) -> dict[str, str]:
    manifest = prediction_path.parent / "query_ids.csv"
    if manifest.exists():
        molecule_ids = pd.read_csv(manifest)["molecule_id"].astype(str)
    else:
        molecule_ids = frame["molecule_id"].astype(str).drop_duplicates()
    return {molecule_id: molecule_id for molecule_id in molecule_ids}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune mass/fingerprint reciprocal-rank fusion on a calibration fold"
    )
    parser.add_argument("--tuning-predictions", type=Path, required=True)
    parser.add_argument("--report-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--grid-size", type=int, default=21)
    args = parser.parse_args()
    tuning = pd.read_parquet(args.tuning_predictions)
    report = pd.read_parquet(args.report_predictions)
    required = {"molecule_id", "inchikey14", "rank", "mass_rank"}
    for label, frame in (("tuning", tuning), ("report", report)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions missing columns: {sorted(missing)}")
    tuning_truth = _truth_from_manifest(args.tuning_predictions, tuning)
    rows: list[dict[str, float]] = []
    for weight in np.linspace(0.0, 1.0, args.grid_size):
        metrics = evaluate_mrr(
            tuning_truth, _blend_predictions(tuning, float(weight), args.rrf_k)
        )
        rows.append({"mass_weight": float(weight), "mrr_at_25": metrics.mrr_at_25})
    tuning_grid = pd.DataFrame(rows)
    best = tuning_grid.sort_values(
        ["mrr_at_25", "mass_weight"], ascending=[False, False], kind="stable"
    ).iloc[0]
    best_weight = float(best.mass_weight)
    report_truth = _truth_from_manifest(args.report_predictions, report)
    report_metrics = evaluate_mrr(
        report_truth, _blend_predictions(report, best_weight, args.rrf_k)
    ).to_dict()
    result = {
        "selected_mass_weight": best_weight,
        "selected_fingerprint_weight": 1.0 - best_weight,
        "rrf_k": args.rrf_k,
        "tuning_mrr_at_25": float(best.mrr_at_25),
        "report": report_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tuning_grid.to_csv(args.output.parent / "fingerprint_blend_grid.csv", index=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
