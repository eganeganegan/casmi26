#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from casmi.retrieval.candidate_db import build_candidate_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact offline candidate structure database")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--exact-mass-column", required=True)
    parser.add_argument("--candidate-id-column")
    parser.add_argument("--inchikey-column", required=True)
    parser.add_argument("--formula-column")
    parser.add_argument("--separator")
    args = parser.parse_args()
    report = build_candidate_database(
        args.input,
        args.output,
        smiles_column=args.smiles_column,
        exact_mass_column=args.exact_mass_column,
        source=args.source,
        candidate_id_column=args.candidate_id_column,
        inchikey_column=args.inchikey_column,
        formula_column=args.formula_column,
        separator=args.separator,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
