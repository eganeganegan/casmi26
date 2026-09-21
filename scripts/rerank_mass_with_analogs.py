#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.chemistry import morgan_fingerprint, tanimoto
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank mass-database candidates using spectral-neighbor structural similarity"
    )
    parser.add_argument("--mass-candidates", type=Path, required=True)
    parser.add_argument("--spectral-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-analogs", type=int, default=25)
    parser.add_argument("--mass-tolerance-ppm", type=float, default=20.0)
    parser.add_argument("--analog-weight", type=float, default=0.85)
    args = parser.parse_args()
    started = time.perf_counter()
    mass_frame = pd.read_parquet(args.mass_candidates)
    spectral_frame = pd.read_parquet(args.spectral_candidates)
    spectral_score_column = "max_cosine" if "max_cosine" in spectral_frame else "score"
    fingerprint_cache: dict[str, np.ndarray | None] = {}

    def fingerprint(smiles: str) -> np.ndarray | None:
        if smiles not in fingerprint_cache:
            try:
                fingerprint_cache[smiles] = morgan_fingerprint(smiles)
            except (ValueError, RuntimeError):
                fingerprint_cache[smiles] = None
        return fingerprint_cache[smiles]

    output_rows: list[dict[str, object]] = []
    for molecule_id, candidates in mass_frame.groupby("molecule_id", sort=False):
        analog_rows = (
            spectral_frame[spectral_frame.molecule_id == molecule_id]
            .sort_values("rank")
            .drop_duplicates("inchikey14")
            .head(args.top_analogs)
        )
        analogs: list[tuple[str, float, np.ndarray]] = []
        for row in analog_rows.itertuples(index=False):
            fp = fingerprint(str(row.smiles))
            if fp is not None:
                analogs.append(
                    (str(row.inchikey14), float(getattr(row, spectral_score_column)), fp)
                )
        ranked_rows: list[dict[str, object]] = []
        for row in candidates.itertuples(index=False):
            candidate_fp = fingerprint(str(row.smiles))
            best_analog_score = 0.0
            best_similarity = 0.0
            best_key: str | None = None
            if candidate_fp is not None:
                for analog_key, spectral_score, analog_fp in analogs:
                    similarity = tanimoto(candidate_fp, analog_fp)
                    combined = spectral_score * similarity
                    if combined > best_analog_score:
                        best_analog_score = combined
                        best_similarity = similarity
                        best_key = analog_key
            mass_score = math.exp(
                -abs(float(row.mass_error_ppm)) / max(args.mass_tolerance_ppm, 1e-12)
            )
            final_score = args.analog_weight * best_analog_score + (1 - args.analog_weight) * mass_score
            values = row._asdict()
            values.update(
                {
                    "mass_score": mass_score,
                    "analog_score": best_analog_score,
                    "analog_tanimoto": best_similarity,
                    "best_analog_inchikey14": best_key,
                    "final_score": final_score,
                }
            )
            ranked_rows.append(values)
        ranked_rows.sort(key=lambda value: float(value["final_score"]), reverse=True)
        for rank, values in enumerate(ranked_rows, start=1):
            values["rank"] = rank
            output_rows.append(values)

    output = pd.DataFrame(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    predictions = (
        output.sort_values(["molecule_id", "rank"])
        .groupby("molecule_id")["inchikey14"]
        .apply(list)
        .to_dict()
    )
    molecule_ids = sorted(set(mass_frame.molecule_id) | set(spectral_frame.molecule_id))
    truth = {str(molecule_id): str(molecule_id) for molecule_id in molecule_ids}
    metrics = evaluate_mrr(truth, predictions).to_dict()
    metrics.update(evaluate_candidate_recall(truth, predictions, (25, 100, 500, 1000, 5000)))
    report = {
        **metrics,
        "analog_weight": args.analog_weight,
        "top_analogs": args.top_analogs,
        "candidate_rows": len(output),
        "fingerprints_cached": len(fingerprint_cache),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output.parent / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
