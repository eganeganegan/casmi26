#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from casmi.validation import candidate_recall_report


def _load_ranked(path: Path) -> dict[str, list[str]]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    rank_col = "rank" if "rank" in frame else None
    if rank_col:
        frame = frame.sort_values(["molecule_id", rank_col])
    if len(frame) and ";" in str(frame.iloc[0]["smiles"]):
        return {
            str(row.molecule_id): str(row.smiles).split(";")
            for row in frame.itertuples(index=False)
        }
    return frame.groupby("molecule_id")["smiles"].apply(list).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate recall across retrieval channels")
    parser.add_argument("--truth", type=Path, required=True, help="CSV with molecule_id and smiles")
    parser.add_argument(
        "--channel",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Ranked parquet/CSV; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth_frame = pd.read_csv(args.truth)
    truth = dict(
        zip(
            truth_frame["molecule_id"].astype(str),
            truth_frame["smiles"],
            strict=True,
        )
    )
    channels: dict[str, dict[str, list[str]]] = {}
    for specification in args.channel:
        name, separator, raw_path = specification.partition("=")
        if not separator:
            parser.error(f"invalid --channel {specification!r}; expected NAME=PATH")
        channels[name] = _load_ranked(Path(raw_path))
    report = candidate_recall_report(truth, channels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
