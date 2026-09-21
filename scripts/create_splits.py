#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from casmi.data.schema import detect_columns
from casmi.validation import create_grouped_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe molecule-level CV folds")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--strategy", choices=["structure", "scaffold", "library"], default="structure")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    lazy = pl.scan_parquet(args.input)
    cmap = detect_columns(lazy.collect_schema().names())
    if not cmap.smiles:
        raise ValueError("Training parquet has no detected SMILES label")
    molecule_expression = (
        pl.col(cmap.molecule_id)
        if cmap.molecule_id
        else pl.col(cmap.inchikey14 or cmap.smiles).alias("molecule_id")
    )
    expressions = [molecule_expression, pl.col(cmap.smiles).alias("normalized_smiles")]
    if cmap.inchikey14:
        expressions.append(pl.col(cmap.inchikey14).alias("inchikey14"))
    frame = lazy.select(expressions).unique().collect(engine="streaming").to_pandas()
    splits = create_grouped_splits(
        frame,
        molecule_col=cmap.molecule_id or "molecule_id",
        smiles_col="normalized_smiles",
        inchikey_col="inchikey14",
        n_splits=args.folds,
        strategy=args.strategy,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(args.output, index=False)
    print(splits.groupby("fold").size().to_string())


if __name__ == "__main__":
    main()
