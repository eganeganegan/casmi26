#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmi.retrieval import normalize_candidate_isotopes


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize isotope-labelled candidate structures")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_candidate_isotopes(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
