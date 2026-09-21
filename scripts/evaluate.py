#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import polars as pl

from casmi.data.schema import detect_columns
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked predictions with structure-level MRR")
    parser.add_argument("--truth", type=Path, required=True, help="Labeled parquet")
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    lazy = pl.scan_parquet(args.truth)
    cmap = detect_columns(lazy.collect_schema().names())
    if not cmap.smiles:
        raise ValueError("Truth parquet has no SMILES column")
    truth_frame = lazy.select(cmap.molecule_id, cmap.smiles).unique().collect().to_pandas()
    truth = dict(
        zip(
            truth_frame[cmap.molecule_id].astype(str),
            truth_frame[cmap.smiles],
            strict=True,
        )
    )
    predictions_frame = pd.read_parquet(args.predictions).sort_values(["molecule_id", "rank"])
    predictions = predictions_frame.groupby("molecule_id")["smiles"].apply(list).to_dict()
    report = evaluate_mrr(truth, predictions).to_dict()
    report.update(evaluate_candidate_recall(truth, predictions, (25, 100, 500)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
