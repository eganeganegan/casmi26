#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from casmi.data.schema import inspect_parquet, report_to_json
from casmi.utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CASMI parquet schema and statistics")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--example-rows", type=int, default=3)
    args = parser.parse_args()
    configure_logging()
    if not args.train and not args.test:
        parser.error("provide --train and/or --test")
    for label, path in (("train", args.train), ("test", args.test)):
        if path:
            if not path.exists():
                raise FileNotFoundError(f"{label} parquet not found: {path}")
            print(f"\n[{label}]\n{report_to_json(inspect_parquet(path, args.example_rows))}")


if __name__ == "__main__":
    main()
