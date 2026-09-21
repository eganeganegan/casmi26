#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from casmi.inference import make_submission, make_submission_from_ranked_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and validate competition submission.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=25)
    args = parser.parse_args()
    frame = pd.read_parquet(args.predictions)
    if "inchikey14" in frame.columns:
        submission = make_submission_from_ranked_frame(
            frame, args.output, max_candidates=args.max_candidates
        )
    else:
        frame = frame.sort_values(["molecule_id", "rank"])
        predictions = frame.groupby("molecule_id")["smiles"].apply(list).to_dict()
        submission = make_submission(predictions, args.output, max_candidates=args.max_candidates)
    print(f"validated and wrote {len(submission):,} molecule rows to {args.output}")


if __name__ == "__main__":
    main()
