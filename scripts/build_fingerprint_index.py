#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmi.retrieval import CandidateDatabase, CandidateFingerprintIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bit-packed Morgan candidate index")
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()
    database = CandidateDatabase.from_parquet(args.candidate_db)
    index, report = CandidateFingerprintIndex.from_candidate_database(
        database, n_bits=args.n_bits, radius=args.radius
    )
    index.save(args.output)
    report.update(
        {
            "n_bits": args.n_bits,
            "radius": args.radius,
            "storage_mb": index.storage_nbytes / (1024**2),
            "output": str(args.output),
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
