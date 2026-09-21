#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
from kagglehub.exceptions import UnauthenticatedError

COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
FILES = ("train.parquet", "test.parquet", "sample_submission.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CASMI competition data with kagglehub")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Replace files already downloaded")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        destination = args.output / filename
        if destination.exists() and not args.force:
            print(f"already present: {destination}")
            continue
        try:
            downloaded = kagglehub.competition_download(
                COMPETITION,
                path=filename,
                output_dir=str(args.output),
                force_download=args.force,
            )
        except UnauthenticatedError as exc:
            raise SystemExit(
                "Kaggle authentication is required. Accept the competition rules, then run "
                "`.venv/bin/python -c \"import kagglehub; kagglehub.login()\"` or set "
                "KAGGLE_API_TOKEN, and retry `make download`."
            ) from exc
        print(f"downloaded {filename}: {downloaded}")

    missing = [str(args.output / filename) for filename in FILES if not (args.output / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Download completed without expected files: {missing}")
    print(f"competition files ready in {args.output.resolve()}")


if __name__ == "__main__":
    main()
