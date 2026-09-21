#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from casmi.chemistry import morgan_fingerprint, neutral_mass_from_precursor
from casmi.models import (
    load_fingerprint_model,
    pool_fingerprint_probabilities,
    require_torch,
    spectrum_fingerprint_features,
)
from casmi.pipeline import CASMIPipeline
from casmi.retrieval import CandidateDatabase, CandidateFingerprintIndex
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted fingerprints within observed-mass candidate pools"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spectral-index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--fingerprint-index", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-molecules", type=int, default=1000)
    parser.add_argument("--max-spectra-per-molecule", type=int, default=16)
    parser.add_argument("--mass-tolerance-ppm", type=float, default=20.0)
    parser.add_argument(
        "--fallback-mass-tolerance-ppm",
        type=float,
        action="append",
        default=[],
        help="Wider mass window; may be repeated and is pruned by predicted fingerprint rank",
    )
    parser.add_argument("--fallback-fingerprint-candidates", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-batch-rows", type=int, default=100_000)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output_batch_rows <= 0:
        raise ValueError("output-batch-rows must be positive")
    started = time.perf_counter()
    torch = require_torch()
    model, feature_config, model_config, model_extra = load_fingerprint_model(args.model)
    report_fold = model_extra.get("report_fold")
    early_fold = model_extra.get("early_stopping_fold")
    allowed_folds = {int(value) for value in (report_fold, early_fold) if value is not None}
    if allowed_folds and args.fold not in allowed_folds:
        raise ValueError(
            f"Model evaluation folds are {sorted(allowed_folds)}, but requested fold {args.fold}"
        )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    pipeline = CASMIPipeline.load(args.spectral_index)
    if pipeline.index is None:
        raise ValueError("Serialized pipeline has no spectral index")
    spectral_index = pipeline.index
    database = CandidateDatabase.from_parquet(args.candidate_db)
    fingerprint_index = CandidateFingerprintIndex.load(args.fingerprint_index)
    if fingerprint_index.n_bits != model_config.n_bits:
        raise ValueError("Model and candidate fingerprint sizes differ")

    splits = pd.read_parquet(args.splits)
    available = sorted(
        splits.loc[splits.fold == args.fold, "inchikey14"].astype(str).unique()
    )
    rng = np.random.default_rng(args.seed)
    selected_keys = (
        sorted(rng.choice(np.asarray(available), args.max_molecules, replace=False).tolist())
        if len(available) > args.max_molecules
        else available
    )
    selected_set = set(selected_keys)
    code_by_key = {
        spectral_index.structure_keys[code]: code
        for code in range(len(spectral_index.structure_keys))
        if spectral_index.structure_keys[code] in selected_set
    }
    missing_index = sorted(selected_set - set(code_by_key))
    if missing_index:
        raise ValueError(f"Selected structures missing from spectral index: {missing_index[:5]}")
    selected_codes = set(code_by_key.values())
    spectra_by_code: dict[int, list[int]] = defaultdict(list)
    for spectrum_index, raw_code in enumerate(spectral_index.structure_codes):
        code = int(raw_code)
        if code in selected_codes and len(spectra_by_code[code]) < args.max_spectra_per_molecule:
            spectra_by_code[code].append(spectrum_index)

    feature_rows: list[np.ndarray] = []
    feature_codes: list[int] = []
    observed_masses: dict[int, list[float]] = defaultdict(list)
    observed_adducts: dict[int, set[str]] = defaultdict(set)
    observed_collision_energies: dict[int, list[float]] = defaultdict(list)
    positive_spectra: dict[int, int] = defaultdict(int)
    for code in sorted(selected_codes):
        for spectrum_index in spectra_by_code[code]:
            start = int(spectral_index.peak_offsets[spectrum_index])
            end = int(spectral_index.peak_offsets[spectrum_index + 1])
            adduct = spectral_index.adduct_values[int(spectral_index.adduct_codes[spectrum_index])]
            polarity_code = int(spectral_index.polarities[spectrum_index])
            ionization_mode = (
                "positive" if polarity_code == 1 else "negative" if polarity_code == -1 else None
            )
            collision_energy = float(spectral_index.collision_energies[spectrum_index])
            precursor_mz = float(spectral_index.precursor_mz[spectrum_index])
            observed_adducts[code].add(adduct)
            if np.isfinite(collision_energy):
                observed_collision_energies[code].append(collision_energy)
            positive_spectra[code] += polarity_code == 1
            feature_rows.append(
                spectrum_fingerprint_features(
                    spectral_index.peak_mz[start:end],
                    spectral_index.peak_intensity[start:end],
                    precursor_mz=precursor_mz,
                    adduct=adduct,
                    collision_energy=collision_energy,
                    ionization_mode=ionization_mode,
                    config=feature_config,
                )
            )
            feature_codes.append(code)
            try:
                mass = neutral_mass_from_precursor(precursor_mz, adduct)
            except ValueError:
                continue
            if np.isfinite(mass) and mass > 0:
                observed_masses[code].append(float(mass))
    features = np.stack(feature_rows)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            batch = torch.from_numpy(features[start : start + args.batch_size]).to(
                device=device, dtype=torch.float32
            )
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    all_probabilities = np.concatenate(probabilities)
    probability_by_code: dict[int, np.ndarray] = {}
    feature_codes_array = np.asarray(feature_codes)
    for code in selected_codes:
        probability_by_code[code] = pool_fingerprint_probabilities(
            all_probabilities[feature_codes_array == code]
        )

    mass_predictions: dict[str, list[str]] = {}
    model_predictions: dict[str, list[str]] = {}
    oracle_predictions: dict[str, list[str]] = {}
    output_rows: list[dict[str, object]] = []
    part_directory: tempfile.TemporaryDirectory[str] | None = None
    part_paths: list[Path] = []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        part_directory = tempfile.TemporaryDirectory(
            prefix=f".{args.output.name}.parts.", dir=args.output.parent
        )

    def flush_output_rows() -> None:
        nonlocal output_rows
        if not output_rows or part_directory is None:
            return
        part_path = Path(part_directory.name) / f"part-{len(part_paths):05d}.parquet"
        pd.DataFrame(output_rows).to_parquet(part_path, index=False)
        part_paths.append(part_path)
        output_rows = []

    missing_mass = 0
    missing_fingerprint = 0
    for key in tqdm(selected_keys, desc="candidate ranking"):
        code = code_by_key[key]
        if not observed_masses[code]:
            mass_predictions[key] = []
            model_predictions[key] = []
            oracle_predictions[key] = []
            missing_mass += 1
            continue
        neutral_mass = float(np.median(observed_masses[code]))
        tolerances = [args.mass_tolerance_ppm, *args.fallback_mass_tolerance_ppm]
        widest_tolerance = max(tolerances)
        candidates = database.query_mass(
            neutral_mass, widest_tolerance, limit=args.max_candidates
        )
        candidate_by_key = {candidate.inchikey14: candidate for candidate in candidates}
        fp_indices = fingerprint_index.indices_for_keys(
            candidate.inchikey14 for candidate in candidates
        )
        if not len(fp_indices):
            mass_predictions[key] = [
                candidate.inchikey14
                for candidate in candidates
                if abs(float(candidate.mass_error_ppm or 0.0)) <= args.mass_tolerance_ppm
            ]
            model_predictions[key] = []
            oracle_predictions[key] = []
            missing_fingerprint += 1
            continue
        candidate_keys = [
            fingerprint_index.keys[int(index)].decode("ascii") for index in fp_indices
        ]
        fingerprint_key_set = set(candidate_keys)
        wide_predicted_hits = fingerprint_index.rank_probabilities(
            probability_by_code[code], candidate_indices=fp_indices, top_n=len(fp_indices)
        )
        primary_keys = {
            candidate.inchikey14
            for candidate in candidates
            if abs(float(candidate.mass_error_ppm or 0.0)) <= args.mass_tolerance_ppm
        }
        fallback_keys = {
            hit.inchikey14
            for hit in wide_predicted_hits[: args.fallback_fingerprint_candidates]
        }
        selected_keys_set = primary_keys | fallback_keys
        selected_candidates = [
            candidate
            for candidate in candidates
            if candidate.inchikey14 in selected_keys_set
            and candidate.inchikey14 in fingerprint_key_set
        ]
        selected_key_set = {candidate.inchikey14 for candidate in selected_candidates}
        predicted_hits = [
            hit for hit in wide_predicted_hits if hit.inchikey14 in selected_key_set
        ]
        if args.skip_oracle:
            wide_oracle_hits = []
            oracle_hits = []
        else:
            truth_probability = morgan_fingerprint(
                spectral_index.structure_smiles[code],
                radius=fingerprint_index.radius,
                n_bits=fingerprint_index.n_bits,
            ).astype(np.float32)
            wide_oracle_hits = fingerprint_index.rank_probabilities(
                truth_probability, candidate_indices=fp_indices, top_n=len(fp_indices)
            )
            oracle_hits = [
                hit for hit in wide_oracle_hits if hit.inchikey14 in selected_key_set
            ]
        mass_predictions[key] = [candidate.inchikey14 for candidate in selected_candidates]
        model_predictions[key] = [hit.inchikey14 for hit in predicted_hits]
        oracle_predictions[key] = [hit.inchikey14 for hit in oracle_hits]
        mass_rank = {
            candidate.inchikey14: rank
            for rank, candidate in enumerate(selected_candidates, 1)
        }
        wide_mass_rank = {
            candidate.inchikey14: rank for rank, candidate in enumerate(candidates, 1)
        }
        wide_fingerprint_rank = {
            hit.inchikey14: rank for rank, hit in enumerate(wide_predicted_hits, 1)
        }
        oracle_score = {hit.inchikey14: hit.similarity for hit in wide_oracle_hits}
        for rank, hit in enumerate(predicted_hits, 1):
            candidate = candidate_by_key[hit.inchikey14]
            absolute_mass_error = abs(float(candidate.mass_error_ppm or 0.0))
            spectrum_count = len(spectra_by_code[code])
            output_rows.append(
                {
                    "molecule_id": key,
                    "rank": rank,
                    "inchikey14": hit.inchikey14,
                    "smiles": candidate.smiles,
                    "formula": candidate.formula,
                    "source": candidate.source,
                    "target": int(hit.inchikey14 == key),
                    "predicted_fingerprint_score": hit.similarity,
                    "oracle_fingerprint_score": oracle_score.get(hit.inchikey14),
                    "mass_rank": mass_rank[hit.inchikey14],
                    "wide_mass_rank": wide_mass_rank[hit.inchikey14],
                    "wide_fingerprint_rank": wide_fingerprint_rank[hit.inchikey14],
                    "mass_error_ppm": candidate.mass_error_ppm,
                    "absolute_mass_error_ppm": absolute_mass_error,
                    "mass_score_20ppm": np.exp(-absolute_mass_error / 20.0),
                    "within_20ppm": int(absolute_mass_error <= 20.0),
                    "within_50ppm": int(absolute_mass_error <= 50.0),
                    "within_1000ppm": int(absolute_mass_error <= 1000.0),
                    "candidate_exact_mass": candidate.exact_mass,
                    "neutral_mass": neutral_mass,
                    "candidate_count": len(selected_candidates),
                    "wide_candidate_count": len(fp_indices),
                    "query_spectrum_count": spectrum_count,
                    "query_adduct_count": len(observed_adducts[code]),
                    "query_positive_fraction": positive_spectra[code] / spectrum_count,
                    "query_mean_collision_energy": (
                        float(np.mean(observed_collision_energies[code]))
                        if observed_collision_energies[code]
                        else None
                    ),
                }
            )
            if len(output_rows) >= args.output_batch_rows:
                flush_output_rows()

    truth = {key: key for key in selected_keys}
    metrics: dict[str, object] = {
        "n_molecules": len(selected_keys),
        "fold": args.fold,
        "seed": args.seed,
        "mass_tolerance_ppm": args.mass_tolerance_ppm,
        "fallback_mass_tolerances_ppm": args.fallback_mass_tolerance_ppm,
        "fallback_fingerprint_candidates": args.fallback_fingerprint_candidates,
        "missing_mass": missing_mass,
        "missing_fingerprint_pool": missing_fingerprint,
        "mass_only": {
            **evaluate_mrr(truth, mass_predictions).to_dict(),
            **evaluate_candidate_recall(truth, mass_predictions, (25, 100, 500, 5000)),
        },
        "predicted_fingerprint": {
            **evaluate_mrr(truth, model_predictions).to_dict(),
            **evaluate_candidate_recall(truth, model_predictions, (25, 100, 500, 5000)),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    if not args.skip_oracle:
        metrics["oracle_fingerprint"] = {
            **evaluate_mrr(truth, oracle_predictions).to_dict(),
            **evaluate_candidate_recall(truth, oracle_predictions, (25, 100, 500, 5000)),
        }
    print(json.dumps(metrics, indent=2))
    if args.output:
        flush_output_rows()
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            dir=args.output.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            if part_paths:
                first = pq.read_table(part_paths[0])
                with pq.ParquetWriter(
                    temporary_path,
                    first.schema,
                    compression="zstd",
                    write_statistics=True,
                ) as writer:
                    writer.write_table(first)
                    for part_path in part_paths[1:]:
                        writer.write_table(pq.read_table(part_path, schema=first.schema))
            else:
                pd.DataFrame(output_rows).to_parquet(temporary_path, index=False)
            temporary_path.replace(args.output)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            if part_directory is not None:
                part_directory.cleanup()
        pd.DataFrame({"molecule_id": selected_keys}).to_csv(
            args.output.parent / "query_ids.csv", index=False
        )
        (args.output.parent / "fingerprint_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
