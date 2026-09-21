#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmi.retrieval import build_numpy_candidate_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate DB from NumPy metadata/mass bundle")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mass", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    report = build_numpy_candidate_database(
        args.metadata, args.mass, args.output, source=args.source
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
