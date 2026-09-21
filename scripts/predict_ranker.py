#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def assign_final_ranks(frame: pd.DataFrame, score_column: str) -> pd.Series:
    required = {"molecule_id", "mass_rank", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    order = frame.sort_values(
        ["molecule_id", score_column, "mass_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranks = pd.Series(index=frame.index, dtype=np.int64)
    ranks.loc[order.index] = (
        order.groupby("molecule_id", sort=False).cumcount().to_numpy() + 1
    )
    return ranks.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen LightGBM candidate ranker")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-columns", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    feature_columns_path = args.feature_columns or args.model.with_name(
        "feature_columns.json"
    )
    columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    if not isinstance(columns, list) or not columns or not all(
        isinstance(column, str) for column in columns
    ):
        raise ValueError("feature_columns.json must contain a non-empty string list")
    frame = pd.read_parquet(args.features)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing model columns: {sorted(missing)}")
    booster = lgb.Booster(model_file=str(args.model))
    if booster.num_feature() != len(columns):
        raise ValueError(
            f"Model expects {booster.num_feature()} features but schema has {len(columns)}"
        )
    frame["ranker_score"] = booster.predict(frame[columns])
    frame["rank"] = assign_final_ranks(frame, "ranker_score")
    ordered = frame.sort_values(["molecule_id", "rank"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_parquet(args.output, index=False)
    print(
        json.dumps(
            {
                "rows": len(ordered),
                "queries": int(ordered["molecule_id"].nunique()),
                "feature_count": len(columns),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
