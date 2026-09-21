#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmi.retrieval import merge_candidate_databases


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and InChIKey14-deduplicate candidate DBs")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_candidate_databases(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
