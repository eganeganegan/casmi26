#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmi.retrieval import build_sdf_candidate_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact candidate DB from SDF/SDF.GZ")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--id-property")
    parser.add_argument("--inchikey-property")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    report = build_sdf_candidate_database(
        args.input,
        args.output,
        source=args.source,
        id_property=args.id_property,
        inchikey_property=args.inchikey_property,
        batch_size=args.batch_size,
        max_records=args.max_records,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
