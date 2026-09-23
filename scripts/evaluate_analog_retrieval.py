#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import rdBase

from casmi.chemistry import morgan_fingerprint
from casmi.data import iter_parquet_spectra
from casmi.pipeline import CASMIPipeline
from casmi.ranking import prepare_analog_ranking_features, score_analog_fingerprint_features
from casmi.retrieval import (
    CandidateFingerprintIndex,
    RawRepresentativeEntropyIndex,
    RepresentativeEntropyIndex,
)
from casmi.spectra import Spectrum
from casmi.validation import evaluate_mrr


def _query_spectra(index: object, selected_keys: list[str], max_spectra: int) -> dict[str, list[Spectrum]]:
    key_to_code = {index.structure_keys[code]: code for code in range(len(index.structure_keys))}
    code_to_key = {key_to_code[key]: key for key in selected_keys if key in key_to_code}
    selected_codes = np.asarray(sorted(code_to_key), dtype=np.uint32)
    spectrum_indices = np.flatnonzero(np.isin(index.structure_codes, selected_codes))
    grouped: dict[int, list[int]] = {code: [] for code in code_to_key}
    for raw_index in spectrum_indices:
        grouped[int(index.structure_codes[raw_index])].append(int(raw_index))
    output: dict[str, list[Spectrum]] = {}
    for code, indices in grouped.items():
        indices.sort(
            key=lambda value: int(index.peak_offsets[value + 1] - index.peak_offsets[value]),
            reverse=True,
        )
        spectra: list[Spectrum] = []
        for spectrum_index in indices[:max_spectra]:
            start = int(index.peak_offsets[spectrum_index])
            end = int(index.peak_offsets[spectrum_index + 1])
            collision_energy = float(index.collision_energies[spectrum_index])
            polarity = int(index.polarities[spectrum_index])
            spectra.append(
                Spectrum(
                    molecule_id=code_to_key[code],
                    spectrum_id=index.spectrum_ids[spectrum_index],
                    precursor_mz=float(index.precursor_mz[spectrum_index]),
                    adduct=index.adduct_values[int(index.adduct_codes[spectrum_index])],
                    collision_energy=(collision_energy if np.isfinite(collision_energy) else None),
                    mz=index.peak_mz[start:end],
                    intensity=index.peak_intensity[start:end],
                    exact_mass=float(index.neutral_masses[spectrum_index]),
                    ionization_mode=(
                        "positive" if polarity > 0 else "negative" if polarity < 0 else None
                    ),
                )
            )
        output[code_to_key[code]] = spectra
    return output


def _raw_query_spectra(
    path: Path, selected_keys: list[str], max_spectra: int
) -> dict[str, list[Spectrum]]:
    selected = set(selected_keys)
    grouped: dict[str, list[Spectrum]] = {key: [] for key in selected_keys}
    for spectrum in iter_parquet_spectra(path, molecule_ids=selected):
        key = str(spectrum.inchikey14 or spectrum.molecule_id)
        if key in grouped:
            grouped[key].append(spectrum)
    for key, spectra in grouped.items():
        spectra.sort(key=lambda value: len(value.mz), reverse=True)
        grouped[key] = spectra[:max_spectra]
    return grouped


def _rank_predictions(frame: pd.DataFrame, score: str) -> dict[str, list[str]]:
    ordered = frame.sort_values(
        ["molecule_id", score, "final_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    return {
        str(molecule_id): group["inchikey14"].astype(str).tolist()
        for molecule_id, group in ordered.groupby("molecule_id", sort=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate structure-disjoint mass-shifted analog propagation"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--analog-index",
        type=Path,
        help="Optional standalone raw-intensity entropy index",
    )
    parser.add_argument(
        "--train-spectra",
        type=Path,
        help="Load raw validation query spectra from this parquet instead of the compact index",
    )
    parser.add_argument("--fingerprint-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Score only query groups containing a ground-truth candidate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-analogs", type=int, default=100)
    parser.add_argument("--max-query-spectra", type=int, default=3)
    parser.add_argument("--mass-window-da", type=float, default=200.0)
    parser.add_argument("--mz-tolerance-da", type=float, default=0.02)
    parser.add_argument(
        "--require-same-polarity",
        action="store_true",
        help="Compare each query spectrum only with representatives of the same polarity",
    )
    parser.add_argument("--rrf-k", type=float, default=60.0)
    args = parser.parse_args()
    started = time.perf_counter()

    all_ids = pd.read_csv(args.query_ids)["molecule_id"].astype(str).to_numpy()
    prediction_columns = set(pq.read_schema(args.predictions).names)
    baseline_rank_column = "final_rank" if "final_rank" in prediction_columns else "rank"
    if args.positive_only:
        labels = pd.read_parquet(args.predictions, columns=["molecule_id", "target"])
        positive = labels.groupby("molecule_id", sort=False)["target"].max().eq(1)
        positive_ids = set(positive[positive].index.astype(str))
        all_ids = np.asarray([value for value in all_ids if value in positive_ids])
    rng = np.random.default_rng(args.seed)
    if args.queries < 0:
        parser.error("--queries must be non-negative")
    selected_ids = (
        all_ids.tolist()
        if args.queries == 0 or args.queries >= len(all_ids)
        else rng.choice(all_ids, size=args.queries, replace=False).tolist()
    )
    pipeline = CASMIPipeline.load(args.model)
    if pipeline.index is None:
        raise ValueError("Serialized pipeline has no compact spectral index")
    analog_index = (
        RawRepresentativeEntropyIndex.load(args.analog_index)
        if args.analog_index
        else RepresentativeEntropyIndex(pipeline.index)
    )
    query_spectra = (
        _raw_query_spectra(args.train_spectra, selected_ids, args.max_query_spectra)
        if args.train_spectra
        else _query_spectra(pipeline.index, selected_ids, max_spectra=args.max_query_spectra)
    )
    missing_queries = sorted(set(selected_ids) - set(query_spectra))
    if missing_queries:
        raise ValueError(f"{len(missing_queries)} selected query structures are absent from the index")

    columns = ["molecule_id", "inchikey14", "smiles", "target", baseline_rank_column]
    frame = pd.read_parquet(
        args.predictions,
        columns=columns,
        filters=[("molecule_id", "in", selected_ids)],
    )
    if baseline_rank_column != "final_rank":
        frame = frame.rename(columns={baseline_rank_column: "final_rank"})
    frame_groups = {
        str(molecule_id): group.copy()
        for molecule_id, group in frame.groupby("molecule_id", sort=False)
    }
    fingerprint_index = CandidateFingerprintIndex.load(args.fingerprint_index)
    fingerprint_lookup = fingerprint_index._ensure_lookup()
    scored_groups: list[pd.DataFrame] = []
    analog_counts: list[int] = []
    with rdBase.BlockLogs():
        for query_number, molecule_id in enumerate(selected_ids, start=1):
            group = frame_groups.get(molecule_id)
            if group is None or group.empty:
                continue
            hits = analog_index.search_molecule(
                query_spectra[molecule_id],
                top_n=args.top_analogs,
                mass_window_da=args.mass_window_da,
                mz_tolerance_da=args.mz_tolerance_da,
                exclude_inchikey14={molecule_id},
                require_same_polarity=args.require_same_polarity,
                inputs_preprocessed=not bool(args.train_spectra),
            )
            analog_fingerprints: list[np.ndarray] = []
            analog_similarities: list[float] = []
            for hit in hits:
                try:
                    analog_fingerprints.append(
                        morgan_fingerprint(
                            hit.smiles,
                            radius=fingerprint_index.radius,
                            n_bits=fingerprint_index.n_bits,
                        )
                    )
                    analog_similarities.append(hit.similarity)
                except (ValueError, RuntimeError):
                    continue
            analog_counts.append(len(analog_fingerprints))
            candidate_rows = group["inchikey14"].astype(str).tolist()
            missing = [key for key in candidate_rows if key not in fingerprint_lookup]
            if missing:
                raise ValueError(
                    f"{len(missing)} candidates for {molecule_id} are absent from fingerprint index"
                )
            candidate_indices = np.asarray(
                [fingerprint_lookup[key] for key in candidate_rows], dtype=np.int64
            )
            candidate_fingerprints = fingerprint_index.unpack(candidate_indices)
            analog_matrix = (
                np.stack(analog_fingerprints)
                if analog_fingerprints
                else np.empty((0, fingerprint_index.n_bits), dtype=np.uint8)
            )
            features = score_analog_fingerprint_features(
                candidate_fingerprints,
                analog_matrix,
                analog_similarities,
            )
            for name, values in features.items():
                group[name] = values
            group = group.sort_values(
                ["analog_score_power4", "final_rank"],
                ascending=[False, True],
                kind="stable",
            )
            group["analog_rank"] = np.arange(1, len(group) + 1)
            scored_groups.append(group)
            if query_number % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f"scored {query_number}/{len(selected_ids)} queries in {elapsed:.1f}s", flush=True)

    scored = pd.concat(scored_groups, ignore_index=True) if scored_groups else frame.iloc[0:0]
    scored = prepare_analog_ranking_features(scored)
    truth = {molecule_id: molecule_id for molecule_id in selected_ids}
    predictions = {
        "baseline": {
            str(molecule_id): group.sort_values("final_rank")["inchikey14"].astype(str).tolist()
            for molecule_id, group in scored.groupby("molecule_id", sort=False)
        },
        "analog_only": _rank_predictions(scored, "analog_score_power4"),
    }
    for analog_weight in np.linspace(0.1, 0.9, 9):
        name = f"rrf_analog_{analog_weight:.1f}"
        scored[name] = (
            (1.0 - analog_weight) / (args.rrf_k + scored["final_rank"].astype(float))
            + analog_weight / (args.rrf_k + scored["analog_rank"].astype(float))
        )
        predictions[name] = _rank_predictions(scored, name)
    report = {
        "queries": len(selected_ids),
        "positive_only": args.positive_only,
        "seed": args.seed,
        "representatives": len(analog_index),
        "representative_structures": analog_index.structure_count,
        "representative_peaks": analog_index.peak_count,
        "top_analogs": args.top_analogs,
        "max_query_spectra": args.max_query_spectra,
        "mass_window_da": args.mass_window_da,
        "require_same_polarity": args.require_same_polarity,
        "median_valid_analogs": float(np.median(analog_counts)) if analog_counts else 0.0,
        "metrics": {name: evaluate_mrr(truth, values).to_dict() for name, values in predictions.items()},
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
