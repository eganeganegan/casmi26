#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import kagglehub
from kagglehub.exceptions import UnauthenticatedError


@dataclass(frozen=True)
class ExternalSource:
    handle: str
    directory: str
    expected_files: tuple[str, ...]
    license_name: str


SOURCES = {
    "coconut": ExternalSource(
        handle="prvsiyan/coconut-casmi26-candidates",
        directory="coconut",
        expected_files=("coconut_csv_lite-09-2026.csv",),
        license_name="CC BY 4.0",
    ),
    "chebi_lipidmaps": ExternalSource(
        handle="prvsiyan/chebi-lipidmaps-casmi26",
        directory="chebi_lipidmaps",
        expected_files=("bio_meta.pkl", "bio_mass.npy", "bio_fp.npy"),
        license_name="CC BY-NC-SA 4.0",
    ),
    "pubchemlite": ExternalSource(
        handle="thedevastator/pubchemlite-compound-collection-for-exposomics-3",
        directory="pubchemlite",
        expected_files=("PubChemLite_31Oct2020_exposomics.csv",),
        license_name="CC0 1.0",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public candidate sources for local database construction"
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        help="Source to download; repeat as needed (default: all)",
    )
    parser.add_argument("--output", type=Path, default=Path("data/external"))
    parser.add_argument("--force", action="store_true", help="Replace an existing source download")
    args = parser.parse_args()

    selected = args.source or list(SOURCES)
    for name in selected:
        source = SOURCES[name]
        destination = args.output / source.directory
        expected = [destination / filename for filename in source.expected_files]
        if all(path.exists() for path in expected) and not args.force:
            print(f"already present: {name} ({source.license_name}) in {destination}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = kagglehub.dataset_download(
                source.handle,
                output_dir=str(destination),
                force_download=args.force,
            )
        except UnauthenticatedError as exc:
            raise SystemExit(
                "Kaggle authentication is required. Run "
                "`.venv/bin/python -c \"import kagglehub; kagglehub.login()\"` "
                "or set KAGGLE_API_TOKEN, then retry."
            ) from exc
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{name} downloaded to {downloaded}, but expected files are missing: {missing}"
            )
        print(f"downloaded {name} ({source.license_name}): {downloaded}")


if __name__ == "__main__":
    main()
