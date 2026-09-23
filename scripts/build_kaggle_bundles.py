#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path, *, force: bool) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the EXP029 source and frozen EXP027 model-asset Kaggle datasets"
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--asset-output", type=Path, default=Path("dist/casmi26-exp027-assets")
    )
    parser.add_argument(
        "--source-output", type=Path, default=Path("dist/casmi26-source/casmi26")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    assets = {
        "expanded_candidates.parquet": root / "data/processed/expanded_candidates.parquet",
        "expanded_morgan_2048.joblib": root / "data/cache/expanded_morgan_2048.joblib",
        "pipeline_compact.joblib": root / "data/cache/pipeline_compact.joblib",
        "raw_entropy_representatives.joblib": root
        / "data/cache/raw_entropy_representatives.joblib",
        "expanded_candidate_descriptors.parquet": root
        / "data/cache/expanded_candidate_descriptors.parquet",
        "fingerprint_model.pt": root / "experiments/EXP018/model.pt",
        "ranker/ranker.txt": root / "experiments/EXP027/ranker/ranker.txt",
        "ranker/feature_columns.json": root
        / "experiments/EXP027/ranker/feature_columns.json",
    }
    manifest = {
        "experiment": "EXP027",
        "candidate_sources": ["COCONUT", "PubChemLite", "ChEMBL 37"],
        "licenses": {
            "COCONUT": "CC BY 4.0",
            "PubChemLite": "CC0 1.0",
            "ChEMBL 37": "CC BY-SA 3.0",
            "CASMI 2026 train-derived spectral index": "CC BY-NC 4.0",
        },
        "artifacts": [],
    }
    for relative, source in assets.items():
        record = _copy(source, args.asset_output / relative, force=args.force)
        record["file"] = relative
        record["source"] = str(source.relative_to(root))
        manifest["artifacts"].append(record)
    (args.asset_output / "ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    source_files = (
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
    )
    for name in source_files:
        _copy(root / name, args.source_output / name, force=args.force)
    wheels = sorted((root / "vendor/wheels").glob("rdkit-2026.3.3-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            "Missing vendor/wheels/rdkit-2026.3.3-*.whl; download CPython 3.11/3.12 "
            "manylinux wheels before building the offline Kaggle source bundle"
        )
    for wheel in wheels:
        _copy(wheel, args.source_output / "wheels" / wheel.name, force=args.force)
    for directory in ("src", "scripts", "configs"):
        destination = args.source_output / directory
        if destination.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --force")
        shutil.copytree(root / directory, destination, dirs_exist_ok=args.force)
    print(
        json.dumps(
            {
                "asset_output": str(args.asset_output),
                "source_output": str(args.source_output),
                "asset_bytes": sum(
                    int(record["bytes"]) for record in manifest["artifacts"]
                ),
                "artifact_count": len(manifest["artifacts"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
