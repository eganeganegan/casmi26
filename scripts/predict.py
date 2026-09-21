#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from casmi.data import iter_parquet_spectra
from casmi.pipeline import CASMIPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run molecule-level spectral retrieval")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--mass-tolerance-ppm", type=float)
    parser.add_argument("--mz-tolerance-da", type=float)
    args = parser.parse_args()
    pipeline = CASMIPipeline.load(args.model)
    if args.mass_tolerance_ppm is not None:
        pipeline.mass_tolerance_ppm = args.mass_tolerance_ppm
    if args.mz_tolerance_da is not None:
        pipeline.mz_tolerance = args.mz_tolerance_da
    predictions = pipeline.predict_dataset(
        iter_parquet_spectra(args.test), max_candidates=args.max_candidates
    )
    rows = []
    for molecule_id, prediction in predictions.items():
        for rank, evidence in enumerate(prediction.evidence, start=1):
            rows.append({"molecule_id": molecule_id, "rank": rank, **evidence.to_dict()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"wrote {len(rows):,} ranked candidates for {len(predictions):,} molecules")


if __name__ == "__main__":
    main()
