#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from casmi.chemistry import neutral_mass_from_precursor
from casmi.retrieval import CandidateDatabase
from casmi.validation import evaluate_candidate_recall, evaluate_mrr, stratified_candidate_recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact-mass candidate database recall")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-validation-molecules", type=int, default=100)
    parser.add_argument("--mass-tolerance-ppm", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    splits = pd.read_parquet(args.splits)
    all_validation_ids = sorted(
        splits.loc[splits.fold == args.fold, "inchikey14"].astype(str).unique()
    )
    if len(all_validation_ids) > args.max_validation_molecules:
        rng = np.random.default_rng(args.seed)
        validation_ids = sorted(
            rng.choice(
                np.asarray(all_validation_ids),
                size=args.max_validation_molecules,
                replace=False,
            ).tolist()
        )
    else:
        validation_ids = all_validation_ids
    spectra = (
        pl.scan_parquet(args.train)
        .filter(pl.col("inchikey14").is_in(validation_ids))
        .select("inchikey14", "precursor_mz", "adduct", "ingest_lib")
        .collect(engine="streaming")
    )
    masses: dict[str, list[float]] = defaultdict(list)
    adducts: dict[str, list[str]] = defaultdict(list)
    libraries: dict[str, set[str]] = defaultdict(set)
    spectrum_counts: dict[str, int] = defaultdict(int)
    unsupported_adduct_rows = 0
    for row in spectra.iter_rows(named=True):
        molecule_id = str(row["inchikey14"])
        spectrum_counts[molecule_id] += 1
        adducts[molecule_id].append(str(row["adduct"]))
        libraries[molecule_id].add(str(row["ingest_lib"]))
        try:
            mass = neutral_mass_from_precursor(row["precursor_mz"], row["adduct"])
        except ValueError:
            unsupported_adduct_rows += 1
            continue
        if np.isfinite(mass) and mass > 0:
            masses[molecule_id].append(float(mass))

    database = CandidateDatabase.from_parquet(args.candidate_db)
    predictions: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    missing_mass = 0
    for molecule_id in validation_ids:
        if not masses[molecule_id]:
            predictions[molecule_id] = []
            missing_mass += 1
            continue
        neutral_mass = float(np.median(masses[molecule_id]))
        candidates = database.query_mass(
            neutral_mass, args.mass_tolerance_ppm, limit=args.max_candidates
        )
        predictions[molecule_id] = [candidate.inchikey14 for candidate in candidates]
        if args.output:
            for rank, candidate in enumerate(candidates, start=1):
                rows.append(
                    {
                        "molecule_id": molecule_id,
                        "neutral_mass": neutral_mass,
                        "rank": rank,
                        "candidate_id": candidate.candidate_id,
                        "smiles": candidate.smiles,
                        "inchikey14": candidate.inchikey14,
                        "formula": candidate.formula,
                        "exact_mass": candidate.exact_mass,
                        "mass_error_ppm": candidate.mass_error_ppm,
                        "source": candidate.source,
                    }
                )
    truth = {molecule_id: molecule_id for molecule_id in validation_ids}
    metrics = evaluate_mrr(truth, predictions).to_dict()
    metrics.update(evaluate_candidate_recall(truth, predictions, (25, 100, 500, 1000, 5000)))
    counts = [len(predictions[molecule_id]) for molecule_id in validation_ids]
    database_truth = set(
        database.frame.filter(pl.col("inchikey14").is_in(validation_ids))["inchikey14"].to_list()
    )
    report = {
        **metrics,
        "database_structures": len(database),
        "truth_present_in_database": len(database_truth),
        "database_coverage": len(database_truth) / len(validation_ids),
        "mass_tolerance_ppm": args.mass_tolerance_ppm,
        "seed": args.seed,
        "missing_neutral_mass": missing_mass,
        "unsupported_adduct_rows": unsupported_adduct_rows,
        "candidate_count_min": min(counts, default=0),
        "candidate_count_median": float(np.median(counts)) if counts else 0.0,
        "candidate_count_mean": float(np.mean(counts)) if counts else 0.0,
        "candidate_count_max": max(counts, default=0),
        "runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.output, index=False)
        (args.output.parent / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        neutral_mass_by_id = {
            molecule_id: float(np.median(values)) if values else float("nan")
            for molecule_id, values in masses.items()
        }
        dominant_adduct = {
            molecule_id: max(set(values), key=values.count) if values else "unknown"
            for molecule_id, values in adducts.items()
        }
        mass_bin = {
            molecule_id: (
                "<250"
                if mass < 250
                else "250-350"
                if mass < 350
                else "350-500"
                if mass < 500
                else ">=500"
            )
            for molecule_id, mass in neutral_mass_by_id.items()
        }
        spectrum_bin = {
            molecule_id: (
                "1"
                if count == 1
                else "2-3"
                if count <= 3
                else "4-10"
                if count <= 10
                else "11+"
            )
            for molecule_id, count in spectrum_counts.items()
        }
        stratified = stratified_candidate_recall(
            truth,
            predictions,
            {"mass_bin": mass_bin, "dominant_adduct": dominant_adduct, "spectra": spectrum_bin},
            ks=(25, 100, 500),
        )
        library_rows: list[dict[str, object]] = []
        all_libraries = sorted(set().union(*libraries.values()))
        for library in all_libraries:
            subset_ids = [molecule_id for molecule_id in validation_ids if library in libraries[molecule_id]]
            subset_truth = {molecule_id: molecule_id for molecule_id in subset_ids}
            subset_predictions = {molecule_id: predictions[molecule_id] for molecule_id in subset_ids}
            for name, value in evaluate_candidate_recall(
                subset_truth, subset_predictions, (25, 100, 500)
            ).items():
                library_rows.append(
                    {
                        "stratum": "ingest_lib_membership",
                        "level": library,
                        "cutoff": int(name.removeprefix("recall_at_")),
                        "recall": value,
                        "n_molecules": len(subset_ids),
                    }
                )
        pd.concat([stratified, pd.DataFrame(library_rows)], ignore_index=True).to_csv(
            args.output.parent / "recall_stratified.csv", index=False
        )

        error_rows: list[dict[str, object]] = []
        for molecule_id in validation_ids:
            keys = predictions[molecule_id]
            rank = keys.index(molecule_id) + 1 if molecule_id in keys else None
            if molecule_id not in database_truth:
                failure = "truth_absent_from_database"
            elif rank is None:
                failure = "truth_missed_by_mass_filter"
            elif rank > 25:
                failure = "truth_ranked_below_25"
            else:
                failure = "hit_at_25"
            error_rows.append(
                {
                    "molecule_id": molecule_id,
                    "truth_rank": rank,
                    "failure_category": failure,
                    "candidate_count": len(keys),
                    "neutral_mass": neutral_mass_by_id.get(molecule_id),
                    "dominant_adduct": dominant_adduct.get(molecule_id, "unknown"),
                    "adducts": ";".join(sorted(set(adducts[molecule_id]))),
                    "num_spectra": spectrum_counts[molecule_id],
                    "ingest_libraries": ";".join(sorted(libraries[molecule_id])),
                }
            )
        pd.DataFrame(error_rows).to_csv(args.output.parent / "error_analysis.csv", index=False)


if __name__ == "__main__":
    main()
