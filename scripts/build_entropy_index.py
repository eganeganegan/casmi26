#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from casmi.retrieval import RawRepresentativeEntropyIndex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone raw-intensity representative entropy index"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument(
        "--per-polarity",
        action="store_true",
        help="Retain the richest representative separately for each ionization polarity",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    index = RawRepresentativeEntropyIndex.from_parquet(
        args.input,
        batch_size=args.batch_size,
        per_polarity=args.per_polarity,
    )
    index.save(args.output)
    report = {
        "representatives": len(index),
        "representative_structures": index.structure_count,
        "per_polarity": args.per_polarity,
        "representative_peaks": index.peak_count,
        "storage_mib": index.storage_nbytes / 1024**2,
        "runtime_seconds": time.perf_counter() - started,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
