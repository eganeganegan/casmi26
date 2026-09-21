#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def analyze_failures(
    candidates: pd.DataFrame,
    query_ids: list[str],
    database_keys: set[str],
    *,
    final_rank_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"molecule_id", "inchikey14", "target", final_rank_column}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate rows are missing columns: {sorted(missing)}")
    if candidates.duplicated(["molecule_id", "inchikey14"]).any():
        raise ValueError("Candidate rows repeat a query/InChIKey14 pair")
    grouped = {str(key): group for key, group in candidates.groupby("molecule_id", sort=False)}
    rows: list[dict[str, object]] = []
    for molecule_id in query_ids:
        group = grouped.get(molecule_id)
        truth = (
            group.loc[group["target"].eq(1)].sort_values(final_rank_column).iloc[0]
            if group is not None and group["target"].eq(1).any()
            else None
        )
        database_present = molecule_id in database_keys
        if not database_present:
            category = "truth_absent_from_database"
        elif truth is None:
            category = "truth_missed_by_candidate_generation"
        elif int(truth[final_rank_column]) > 25:
            category = "truth_retrieved_below_25"
        else:
            category = "hit_at_25"
        best_wrong = None
        if group is not None:
            wrong = group.loc[group["target"].eq(0)].sort_values(final_rank_column)
            if len(wrong):
                best_wrong = str(wrong.iloc[0]["inchikey14"])
        row: dict[str, object] = {
            "molecule_id": molecule_id,
            "failure_category": category,
            "database_present": database_present,
            "candidate_count": 0 if group is None else len(group),
            "truth_rank": None if truth is None else int(truth[final_rank_column]),
            "best_wrong_candidate": best_wrong,
        }
        for column in (
            "mass_rank",
            "fingerprint_rank",
            "wide_mass_rank",
            "wide_fingerprint_rank",
            "fragment_intensity_fraction_mean",
        ):
            if column in candidates:
                row[f"truth_{column}"] = None if truth is None else float(truth[column])
        rows.append(row)
    detail = pd.DataFrame(rows)
    counts = detail["failure_category"].value_counts().to_dict()
    summary: dict[str, object] = {
        "queries": len(query_ids),
        "database_present": int(detail["database_present"].sum()),
        "database_coverage": float(detail["database_present"].mean()),
        "categories": {
            category: {
                "count": int(count),
                "fraction": float(count / len(detail)) if len(detail) else 0.0,
            }
            for category, count in counts.items()
        },
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify candidate generation/ranking failures")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--final-rank-column")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = pd.read_parquet(args.candidates)
    query_ids = pd.read_csv(args.query_manifest)["molecule_id"].astype(str).tolist()
    database_table = pq.read_table(args.candidate_db, columns=["inchikey14"])
    database_keys = set(database_table.column("inchikey14").to_pylist())
    rank_column = args.final_rank_column or (
        "final_rank" if "final_rank" in candidates else "rank"
    )
    detail, summary = analyze_failures(
        candidates,
        query_ids,
        database_keys,
        final_rank_column=rank_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
