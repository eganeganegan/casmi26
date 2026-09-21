#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import polars as pl

from casmi.ranking import DESCRIPTOR_FEATURES, prepare_ranking_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize leakage-safe candidate ranking features")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-descriptors", action="store_true")
    parser.add_argument("--descriptor-cache", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.descriptor_cache is not None and not args.no_descriptors:
        candidates = (
            pl.scan_parquet(args.candidates)
            .join(
                pl.scan_parquet(args.descriptor_cache).select(
                    "inchikey14", *DESCRIPTOR_FEATURES
                ),
                on="inchikey14",
                how="left",
            )
            .collect(engine="streaming")
            .to_pandas()
        )
    else:
        candidates = pd.read_parquet(args.candidates)
    features = prepare_ranking_features(
        candidates,
        add_descriptors=not args.no_descriptors,
        descriptor_cache=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    query_manifest = args.candidates.parent / "query_ids.csv"
    destination_manifest = args.output.parent / "query_ids.csv"
    if query_manifest.exists() and query_manifest.resolve() != destination_manifest.resolve():
        shutil.copyfile(query_manifest, destination_manifest)
    report = {
        "rows": len(features),
        "queries": int(features["molecule_id"].nunique()),
        "positive_rows": int(features["target"].sum()),
        "positive_queries": int(features.groupby("molecule_id")["target"].max().sum()),
        "runtime_seconds": time.perf_counter() - started,
        "descriptors": not args.no_descriptors,
        "descriptor_cache": str(args.descriptor_cache) if args.descriptor_cache else None,
    }
    (args.output.parent / f"{args.output.stem}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
