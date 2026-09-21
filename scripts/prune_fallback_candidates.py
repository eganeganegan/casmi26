#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from casmi.retrieval import prune_wide_fallback_candidates
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def _predictions(frame: pd.DataFrame, rank_column: str) -> dict[str, list[str]]:
    return (
        frame.sort_values(["molecule_id", rank_column], kind="stable")
        .groupby("molecule_id", sort=False)["inchikey14"]
        .apply(list)
        .to_dict()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune an over-generated wide-mass fallback pool")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-wide-fingerprint-rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = pd.read_parquet(args.input)
    output = prune_wide_fallback_candidates(
        candidates,
        max_wide_fingerprint_rank=args.max_wide_fingerprint_rank,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    source_manifest = args.input.parent / "query_ids.csv"
    destination_manifest = args.output.parent / "query_ids.csv"
    if source_manifest.exists() and source_manifest.resolve() != destination_manifest.resolve():
        shutil.copyfile(source_manifest, destination_manifest)
    query_ids = (
        pd.read_csv(source_manifest)["molecule_id"].astype(str).tolist()
        if source_manifest.exists()
        else output["molecule_id"].astype(str).drop_duplicates().tolist()
    )
    truth = {molecule_id: molecule_id for molecule_id in query_ids}
    mass_predictions = _predictions(output, "mass_rank")
    fingerprint_predictions = _predictions(output, "rank")
    report = {
        "rows_before": len(candidates),
        "rows_after": len(output),
        "queries": len(query_ids),
        "positive_rows": int(output["target"].sum()) if "target" in output else None,
        "max_wide_fingerprint_rank": args.max_wide_fingerprint_rank,
        "mass_only": {
            **evaluate_mrr(truth, mass_predictions).to_dict(),
            **evaluate_candidate_recall(truth, mass_predictions, (25, 100, 500, 5000)),
        },
        "predicted_fingerprint": {
            **evaluate_mrr(truth, fingerprint_predictions).to_dict(),
            **evaluate_candidate_recall(
                truth, fingerprint_predictions, (25, 100, 500, 5000)
            ),
        },
    }
    (args.output.parent / f"{args.output.stem}_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
