#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi.ranking import fuse_rankings, fuse_scores
from casmi.utils.config import load_config
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def _specifications(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name or not setting:
            raise ValueError(f"Invalid {label} {value!r}; expected NAME=VALUE")
        result[name] = setting
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse ranked candidate channels")
    parser.add_argument("--config", type=Path, default=Path("configs/ensemble.yaml"))
    parser.add_argument("--channel", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--rank-column", action="append", default=[], metavar="NAME=COLUMN")
    parser.add_argument("--score-column", action="append", default=[], metavar="NAME=COLUMN")
    parser.add_argument("--method", choices=("rrf", "borda", "weighted_score"))
    parser.add_argument("--normalization", choices=("zscore", "minmax", "none"), default="zscore")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--truth-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    channels = _specifications(args.channel, "channel")
    rank_columns = _specifications(args.rank_column, "rank column")
    score_columns = _specifications(args.score_column, "score column")
    method = args.method or str(config.get("method", "rrf")).replace(
        "reciprocal_rank_fusion", "rrf"
    )
    weights = {str(key): float(value) for key, value in config.get("weights", {}).items()}

    rankings: dict[str, dict[str, list[str]]] = {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    smiles_by_key: dict[str, str] = {}
    query_ids: set[str] = set()
    for name, path in channels.items():
        frame = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
        rank_column = rank_columns.get(
            name, "final_rank" if "final_rank" in frame.columns else "rank"
        )
        required = {"molecule_id", "inchikey14", "smiles", rank_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Channel {name!r} is missing columns: {sorted(missing)}")
        frame = frame.sort_values(["molecule_id", rank_column], kind="stable")
        rankings[name] = (
            frame.groupby("molecule_id", sort=False)["inchikey14"].apply(list).to_dict()
        )
        if name in score_columns:
            score_column = score_columns[name]
            if score_column not in frame:
                raise ValueError(f"Channel {name!r} has no score column {score_column!r}")
            scores[name] = {
                str(molecule_id): dict(
                    zip(
                        group["inchikey14"].astype(str),
                        group[score_column].astype(float),
                        strict=False,
                    )
                )
                for molecule_id, group in frame.groupby("molecule_id", sort=False)
            }
        smiles_by_key.update(
            zip(frame["inchikey14"].astype(str), frame["smiles"].astype(str), strict=False)
        )
        query_ids.update(frame["molecule_id"].astype(str))
    if args.truth_manifest:
        query_ids.update(pd.read_csv(args.truth_manifest)["molecule_id"].astype(str))

    rows: list[dict[str, object]] = []
    predictions: dict[str, list[str]] = {}
    for molecule_id in sorted(query_ids):
        if method == "weighted_score":
            missing_scores = set(channels) - set(scores)
            if missing_scores:
                raise ValueError(
                    f"Weighted score fusion needs --score-column for {sorted(missing_scores)}"
                )
            fused = fuse_scores(
                {
                    name: values.get(molecule_id, {})
                    for name, values in scores.items()
                },
                weights=weights,
                normalization=args.normalization,
                limit=args.limit,
            )
        else:
            fused = fuse_rankings(
                {name: values.get(molecule_id, []) for name, values in rankings.items()},
                method=method,
                weights=weights,
                rrf_k=float(config.get("k", 60.0)),
                limit=args.limit,
            )
        predictions[molecule_id] = [candidate.candidate for candidate in fused]
        for rank, candidate in enumerate(fused, start=1):
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "rank": rank,
                    "inchikey14": candidate.candidate,
                    "smiles": smiles_by_key[candidate.candidate],
                    "ensemble_score": candidate.score,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    report: dict[str, object] = {
        "method": method,
        "normalization": args.normalization if method == "weighted_score" else None,
        "weights": {name: weights.get(name, 1.0) for name in channels},
        "channels": list(channels),
        "queries": len(query_ids),
        "rows": len(rows),
    }
    if args.truth_manifest:
        truth = {molecule_id: molecule_id for molecule_id in sorted(query_ids)}
        report.update(evaluate_mrr(truth, predictions).to_dict())
        report.update(evaluate_candidate_recall(truth, predictions, (25,)))
    (args.output.parent / f"{args.output.stem}_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
