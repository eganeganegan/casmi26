#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from casmi.data import iter_parquet_spectra
from casmi.pipeline import CASMIPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mass-sorted spectral library index")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    pipeline = CASMIPipeline().fit(iter_parquet_spectra(args.input, args.batch_size))
    pipeline.save(args.output)
    assert pipeline.index is not None
    print(
        f"saved compact pipeline with {len(pipeline.index):,} spectra and "
        f"{pipeline.index.peak_count:,} peaks to {args.output}"
    )


if __name__ == "__main__":
    main()
