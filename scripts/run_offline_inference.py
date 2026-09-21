#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = DEFAULT_PROJECT_ROOT / "src"
if str(DEFAULT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SOURCE_ROOT))

from casmi.inference import validate_submission  # noqa: E402


def build_commands(
    *,
    python: str,
    project_root: Path,
    test: Path,
    candidate_db: Path,
    fingerprint_model: Path,
    fingerprint_index: Path,
    spectral_index: Path,
    analog_index: Path | None,
    descriptor_cache: Path,
    ranker_dir: Path,
    work_dir: Path,
    output: Path,
) -> list[tuple[str, list[str]]]:
    scripts = project_root / "scripts"
    probabilities = work_dir / "fingerprint_probabilities.parquet"
    candidates = work_dir / "candidates.parquet"
    fragments = work_dir / "candidates_fragment.parquet"
    features = work_dir / "features.parquet"
    analog_features = work_dir / "features_analog.parquet"
    derived_analog_features = (
        work_dir / "features_analog_derived.parquet" if analog_index else analog_features
    )
    raw_analog_features = work_dir / "features_analog_raw.parquet"
    ranked = work_dir / "ranked.parquet"
    spectral = work_dir / "spectral.parquet"
    blended = work_dir / "blended.parquet"
    commands = [
        (
            "fingerprint_prediction",
            [
                python,
                str(scripts / "predict_fingerprint.py"),
                "--input",
                str(test),
                "--model",
                str(fingerprint_model),
                "--output",
                str(probabilities),
            ],
        ),
        (
            "candidate_generation",
            [
                python,
                str(scripts / "rank_fingerprint_candidates.py"),
                "--spectra",
                str(test),
                "--probabilities",
                str(probabilities),
                "--candidate-db",
                str(candidate_db),
                "--fingerprint-index",
                str(fingerprint_index),
                "--output",
                str(candidates),
            ],
        ),
        (
            "fragment_scoring",
            [
                python,
                str(scripts / "score_fragment_candidates.py"),
                "--config",
                str(project_root / "configs" / "fragmentation.yaml"),
                "--candidates",
                str(candidates),
                "--spectra",
                str(test),
                "--output",
                str(fragments),
            ],
        ),
        (
            "ranking_features",
            [
                python,
                str(scripts / "build_ranking_dataset.py"),
                "--candidates",
                str(fragments),
                "--descriptor-cache",
                str(descriptor_cache),
                "--output",
                str(features),
            ],
        ),
        (
            "analog_scoring",
            [
                python,
                str(scripts / "score_analog_candidates.py"),
                "--candidates",
                str(features),
                "--spectra",
                str(test),
                "--spectral-index",
                str(spectral_index),
                "--fingerprint-index",
                str(fingerprint_index),
                "--output",
                str(derived_analog_features),
                "--top-analogs",
                "100",
                "--max-query-spectra",
                "3",
                "--mass-window-da",
                "200",
                "--mz-tolerance-da",
                "0.02",
                "--features-only",
            ],
        ),
        (
            "ranker_prediction",
            [
                python,
                str(scripts / "predict_ranker.py"),
                "--features",
                str(analog_features),
                "--model",
                str(ranker_dir / "ranker.txt"),
                "--feature-columns",
                str(ranker_dir / "feature_columns.json"),
                "--output",
                str(ranked),
            ],
        ),
        (
            "spectral_retrieval",
            [
                python,
                str(scripts / "predict.py"),
                "--test",
                str(test),
                "--model",
                str(spectral_index),
                "--output",
                str(spectral),
                "--max-candidates",
                "25",
                "--mass-tolerance-ppm",
                "8.5",
                "--mz-tolerance-da",
                "0.02",
            ],
        ),
        (
            "spectral_gate",
            [
                python,
                str(scripts / "blend_spectral_predictions.py"),
                "--ranked",
                str(ranked),
                "--spectral",
                str(spectral),
                "--output",
                str(blended),
                "--min-cosine",
                "0.95",
                "--min-explained-intensity",
                "0.70",
                "--min-matched-peaks",
                "6",
            ],
        ),
        (
            "submission",
            [
                python,
                str(scripts / "make_submission.py"),
                "--predictions",
                str(blended),
                "--output",
                str(output),
                "--max-candidates",
                "25",
            ],
        ),
    ]
    if analog_index is not None:
        analog_position = next(
            index for index, (name, _) in enumerate(commands) if name == "analog_scoring"
        )
        commands[analog_position + 1 : analog_position + 1] = [
            (
                "raw_analog_scoring",
                [
                    python,
                    str(scripts / "score_analog_candidates.py"),
                    "--candidates",
                    str(features),
                    "--spectra",
                    str(test),
                    "--spectral-index",
                    str(spectral_index),
                    "--analog-index",
                    str(analog_index),
                    "--fingerprint-index",
                    str(fingerprint_index),
                    "--output",
                    str(raw_analog_features),
                    "--top-analogs",
                    "100",
                    "--max-query-spectra",
                    "3",
                    "--mass-window-da",
                    "200",
                    "--mz-tolerance-da",
                    "0.02",
                    "--features-only",
                ],
            ),
            (
                "analog_merge",
                [
                    python,
                    str(scripts / "merge_analog_features.py"),
                    "--features",
                    str(derived_analog_features),
                    "--analog-features",
                    str(raw_analog_features),
                    "--prefix",
                    "raw_",
                    "--output",
                    str(analog_features),
                ],
            ),
        ]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete internet-free CASMI pipeline")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--fingerprint-model", type=Path, required=True)
    parser.add_argument("--fingerprint-index", type=Path, required=True)
    parser.add_argument("--spectral-index", type=Path, required=True)
    parser.add_argument("--analog-index", type=Path)
    parser.add_argument("--descriptor-cache", type=Path, required=True)
    parser.add_argument("--ranker-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("predictions/kaggle"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    if args.output.name != "submission.csv":
        raise ValueError("Kaggle requires the output filename to be submission.csv")
    required_paths = [
        args.project_root / "scripts",
        args.project_root / "src",
        args.test,
        args.candidate_db,
        args.fingerprint_model,
        args.fingerprint_index,
        args.spectral_index,
        args.descriptor_cache,
        args.ranker_dir / "ranker.txt",
        args.ranker_dir / "feature_columns.json",
    ]
    if args.analog_index is not None:
        required_paths.append(args.analog_index)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Offline inference inputs are missing: {missing}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_root = str(args.project_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    timings: dict[str, float] = {}
    started = time.perf_counter()
    for name, command in build_commands(
        python=sys.executable,
        project_root=args.project_root,
        test=args.test,
        candidate_db=args.candidate_db,
        fingerprint_model=args.fingerprint_model,
        fingerprint_index=args.fingerprint_index,
        spectral_index=args.spectral_index,
        analog_index=args.analog_index,
        descriptor_cache=args.descriptor_cache,
        ranker_dir=args.ranker_dir,
        work_dir=args.work_dir,
        output=args.output,
    ):
        step_started = time.perf_counter()
        subprocess.run(
            command,
            cwd=args.project_root,
            env=environment,
            check=True,
        )
        timings[name] = time.perf_counter() - step_started
    submission = pd.read_csv(args.output)
    expected_ids = None
    if args.sample_submission is not None:
        expected_ids = pd.read_csv(args.sample_submission)["molecule_id"].astype(str).tolist()
    validate_submission(submission, expected_ids)
    print(
        json.dumps(
            {
                "status": "ok",
                "submission": str(args.output),
                "rows": len(submission),
                "runtime_seconds": time.perf_counter() - started,
                "step_runtime_seconds": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
