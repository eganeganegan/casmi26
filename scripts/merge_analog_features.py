#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from casmi.ranking import ANALOG_RANKING_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Join leakage-safe analog features to ranking rows")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--analog-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix added to imported analog feature names for multi-channel models",
    )
    args = parser.parse_args()
    base = pd.read_parquet(args.features)
    analog_columns = ["molecule_id", "inchikey14", *ANALOG_RANKING_FEATURES]
    analog = pd.read_parquet(args.analog_features, columns=analog_columns)
    if analog.duplicated(["molecule_id", "inchikey14"]).any():
        raise ValueError("Analog feature rows repeat a query/InChIKey14 pair")
    rename = {column: f"{args.prefix}{column}" for column in ANALOG_RANKING_FEATURES}
    analog = analog.rename(columns=rename)
    output_columns = list(rename.values())
    overlap = set(output_columns) & set(base.columns)
    if overlap:
        base = base.drop(columns=sorted(overlap))
    output = base.merge(
        analog,
        on=["molecule_id", "inchikey14"],
        how="left",
        validate="one_to_one",
    )
    for column in output_columns:
        output[column] = output[column].fillna(0.0).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    source_manifest = args.features.parent / "query_ids.csv"
    destination_manifest = args.output.parent / "query_ids.csv"
    if source_manifest.exists() and source_manifest.resolve() != destination_manifest.resolve():
        shutil.copyfile(source_manifest, destination_manifest)
    print(
        json.dumps(
            {
                "rows": len(output),
                "queries": int(output["molecule_id"].nunique()),
                "analog_queries": int(analog["molecule_id"].nunique()),
                "feature_count": len(output_columns),
                "prefix": args.prefix,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
