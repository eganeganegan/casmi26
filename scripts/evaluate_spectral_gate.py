#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi.validation import evaluate_mrr


def _rankings(frame: pd.DataFrame) -> dict[str, list[str]]:
    rank_column = "final_rank" if "final_rank" in frame.columns else "rank"
    ordered = frame.sort_values(["molecule_id", rank_column], kind="stable")
    return {
        str(molecule_id): group["inchikey14"].astype(str).tolist()
        for molecule_id, group in ordered.groupby("molecule_id", sort=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence-gated library hits ahead of frozen rankings"
    )
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-explained-intensity", type=float, default=0.70)
    parser.add_argument("--min-matched-peaks", type=int, default=6)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        dest="thresholds",
        help="Cosine threshold to report; repeat as needed",
    )
    args = parser.parse_args()
    thresholds = args.thresholds or [0.60, 0.70, 0.80, 0.90, 0.95]
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        parser.error("--threshold values must be between zero and one")

    query_ids = pd.read_csv(args.query_ids)["molecule_id"].astype(str).tolist()
    truth = {molecule_id: molecule_id for molecule_id in query_ids}
    baseline = _rankings(pd.read_parquet(args.ranked))
    spectral = pd.read_csv(args.spectral)
    spectral_by_query = {
        str(row.truth_inchikey14): row for row in spectral.itertuples(index=False)
    }
    report: dict[str, object] = {
        "queries": len(query_ids),
        "min_explained_intensity": args.min_explained_intensity,
        "min_matched_peaks": args.min_matched_peaks,
        "baseline": evaluate_mrr(truth, baseline).to_dict(),
        "thresholds": [],
    }
    for threshold in thresholds:
        predictions: dict[str, list[str]] = {}
        gated_queries = 0
        correct_gates = 0
        for molecule_id in query_ids:
            candidates = list(baseline.get(molecule_id, ()))
            row = spectral_by_query.get(molecule_id)
            if (
                row is not None
                and row.top_max_cosine >= threshold
                and row.top_explained_intensity >= args.min_explained_intensity
                and row.top_matched_peaks >= args.min_matched_peaks
            ):
                top_key = str(row.top_inchikey14)
                candidates = [top_key, *(key for key in candidates if key != top_key)]
                gated_queries += 1
                correct_gates += int(top_key == molecule_id)
            predictions[molecule_id] = candidates
        report["thresholds"].append(
            {
                "min_cosine": threshold,
                "gated_queries": gated_queries,
                "gate_precision": (
                    correct_gates / gated_queries if gated_queries else None
                ),
                **evaluate_mrr(truth, predictions).to_dict(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
