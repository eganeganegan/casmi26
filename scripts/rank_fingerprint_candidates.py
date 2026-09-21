#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from casmi.chemistry import neutral_mass_from_precursor
from casmi.data import iter_parquet_spectra
from casmi.retrieval import CandidateDatabase, CandidateFingerprintIndex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank observed-mass candidates with predicted Morgan probabilities"
    )
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--fingerprint-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mass-tolerance-ppm", type=float, default=20.0)
    parser.add_argument("--empty-fallback-ppm", type=float, default=1000.0)
    parser.add_argument("--max-candidates", type=int, default=5000)
    args = parser.parse_args()

    masses: dict[str, list[float]] = defaultdict(list)
    adducts: dict[str, set[str]] = defaultdict(set)
    collision_energies: dict[str, list[float]] = defaultdict(list)
    positive_spectra: dict[str, int] = defaultdict(int)
    spectrum_counts: dict[str, int] = defaultdict(int)
    for spectrum in iter_parquet_spectra(args.spectra):
        molecule_id = spectrum.molecule_id
        spectrum_counts[molecule_id] += 1
        adducts[molecule_id].add(spectrum.adduct)
        polarity = str(spectrum.ionization_mode or "").lower()
        positive_spectra[molecule_id] += polarity.startswith("pos")
        try:
            collision_energy = float(spectrum.collision_energy)
        except (TypeError, ValueError):
            collision_energy = float("nan")
        if np.isfinite(collision_energy):
            collision_energies[molecule_id].append(collision_energy)
        try:
            mass = neutral_mass_from_precursor(spectrum.precursor_mz, spectrum.adduct)
        except ValueError:
            continue
        if np.isfinite(mass) and mass > 0:
            masses[molecule_id].append(float(mass))
    probability_frame = pd.read_parquet(args.probabilities)
    required = {"molecule_id", "fingerprint_probability"}
    missing = required - set(probability_frame.columns)
    if missing:
        raise ValueError(f"Probability file is missing columns: {sorted(missing)}")
    database = CandidateDatabase.from_parquet(args.candidate_db)
    fingerprint_index = CandidateFingerprintIndex.load(args.fingerprint_index)
    rows: list[dict[str, object]] = []
    fallback_queries = 0
    missing_queries = 0
    for probability_row in tqdm(
        probability_frame.itertuples(index=False),
        total=len(probability_frame),
        desc="candidate ranking",
    ):
        molecule_id = str(probability_row.molecule_id)
        if not masses[molecule_id]:
            missing_queries += 1
            continue
        neutral_mass = float(np.median(masses[molecule_id]))
        candidates = database.query_mass(
            neutral_mass, args.mass_tolerance_ppm, limit=args.max_candidates
        )
        if not candidates and args.empty_fallback_ppm > args.mass_tolerance_ppm:
            candidates = database.query_mass(
                neutral_mass, args.empty_fallback_ppm, limit=args.max_candidates
            )
            fallback_queries += 1
        by_key = {candidate.inchikey14: candidate for candidate in candidates}
        fp_indices = fingerprint_index.indices_for_keys(by_key)
        hits = fingerprint_index.rank_probabilities(
            np.asarray(probability_row.fingerprint_probability, dtype=np.float32),
            candidate_indices=fp_indices,
            top_n=len(fp_indices),
        )
        if not hits:
            missing_queries += 1
            continue
        mass_rank = {
            candidate.inchikey14: rank for rank, candidate in enumerate(candidates, 1)
        }
        spectrum_count = spectrum_counts[molecule_id]
        for rank, hit in enumerate(hits, 1):
            candidate = by_key[hit.inchikey14]
            absolute_mass_error = abs(float(candidate.mass_error_ppm or 0.0))
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "rank": rank,
                    "smiles": candidate.smiles,
                    "inchikey14": candidate.inchikey14,
                    "formula": candidate.formula,
                    "target": 0,
                    "fingerprint_score": hit.similarity,
                    "predicted_fingerprint_score": hit.similarity,
                    "mass_rank": mass_rank[hit.inchikey14],
                    "wide_mass_rank": mass_rank[hit.inchikey14],
                    "wide_fingerprint_rank": rank,
                    "mass_error_ppm": candidate.mass_error_ppm,
                    "absolute_mass_error_ppm": absolute_mass_error,
                    "mass_score_20ppm": np.exp(-absolute_mass_error / 20.0),
                    "within_20ppm": int(absolute_mass_error <= 20.0),
                    "within_50ppm": int(absolute_mass_error <= 50.0),
                    "within_1000ppm": int(absolute_mass_error <= 1000.0),
                    "candidate_exact_mass": candidate.exact_mass,
                    "neutral_mass": neutral_mass,
                    "candidate_count": len(hits),
                    "wide_candidate_count": len(hits),
                    "query_spectrum_count": spectrum_count,
                    "query_adduct_count": len(adducts[molecule_id]),
                    "query_positive_fraction": positive_spectra[molecule_id] / spectrum_count,
                    "query_mean_collision_energy": (
                        float(np.mean(collision_energies[molecule_id]))
                        if collision_energies[molecule_id]
                        else None
                    ),
                    "source": candidate.source,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(
        f"wrote {len(rows):,} fingerprint-ranked candidates for "
        f"{probability_frame['molecule_id'].nunique():,} molecules; "
        f"fallback queries={fallback_queries}, missing queries={missing_queries}"
    )


if __name__ == "__main__":
    main()
