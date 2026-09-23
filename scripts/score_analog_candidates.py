#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase
from tqdm import tqdm

from casmi.chemistry import morgan_fingerprint
from casmi.data import iter_parquet_spectra
from casmi.pipeline import CASMIPipeline
from casmi.ranking import (
    fuse_analog_ranks,
    prepare_analog_ranking_features,
    score_analog_fingerprint_features,
)
from casmi.retrieval import (
    CandidateFingerprintIndex,
    RawRepresentativeEntropyIndex,
    RepresentativeEntropyIndex,
)
from casmi.spectra import Spectrum


def _load_query_spectra(
    path: Path, query_ids: set[str], max_spectra: int
) -> dict[str, list[Spectrum]]:
    grouped: dict[str, list[Spectrum]] = {query_id: [] for query_id in query_ids}
    for spectrum in iter_parquet_spectra(path, molecule_ids=query_ids):
        molecule_id = str(spectrum.molecule_id)
        if molecule_id in grouped:
            grouped[molecule_id].append(spectrum)
    for molecule_id, spectra in grouped.items():
        spectra.sort(key=lambda spectrum: len(spectrum.mz), reverse=True)
        grouped[molecule_id] = spectra[:max_spectra]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add mass-shifted entropy analog evidence to candidate rows"
    )
    parser.add_argument("--ranked", "--candidates", dest="candidates", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--spectral-index", type=Path, required=True)
    parser.add_argument(
        "--analog-index",
        type=Path,
        help="Optional standalone raw-intensity entropy index",
    )
    parser.add_argument("--fingerprint-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-analogs", type=int, default=100)
    parser.add_argument("--max-query-spectra", type=int, default=3)
    parser.add_argument("--mass-window-da", type=float, default=200.0)
    parser.add_argument("--mz-tolerance-da", type=float, default=0.02)
    parser.add_argument(
        "--require-same-polarity",
        action="store_true",
        help="Compare each query spectrum only with representatives of the same polarity",
    )
    parser.add_argument("--analog-weight", type=float, default=0.1)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Write analog features without applying the legacy EXP025 rank fusion",
    )
    args = parser.parse_args()
    if args.top_analogs <= 0 or args.max_query_spectra <= 0:
        parser.error("--top-analogs and --max-query-spectra must be positive")
    started = time.perf_counter()
    ranked = pd.read_parquet(args.candidates)
    required = {"molecule_id", "inchikey14", "smiles", "rank"}
    missing = required - set(ranked.columns)
    if missing:
        raise ValueError(f"Ranked candidates are missing columns: {sorted(missing)}")
    if ranked.duplicated(["molecule_id", "inchikey14"]).any():
        raise ValueError("Ranked candidates repeat a query/InChIKey14 pair")
    query_ids = set(ranked["molecule_id"].astype(str))
    spectra = _load_query_spectra(args.spectra, query_ids, args.max_query_spectra)
    missing_spectra = sorted(key for key, values in spectra.items() if not values)
    if missing_spectra:
        raise ValueError(f"Queries missing spectra: {missing_spectra[:5]}")

    if args.analog_index:
        analog_index = RawRepresentativeEntropyIndex.load(args.analog_index)
    else:
        pipeline = CASMIPipeline.load(args.spectral_index)
        if pipeline.index is None:
            raise ValueError("Serialized pipeline has no compact spectral index")
        analog_index = RepresentativeEntropyIndex(pipeline.index)
    fingerprint_index = CandidateFingerprintIndex.load(args.fingerprint_index)
    fingerprint_lookup = fingerprint_index._ensure_lookup()
    groups: list[pd.DataFrame] = []
    valid_analog_counts: list[int] = []
    with rdBase.BlockLogs():
        iterator = ranked.groupby("molecule_id", sort=False)
        for molecule_id, group in tqdm(iterator, total=len(query_ids), desc="analog scoring"):
            molecule_id = str(molecule_id)
            hits = analog_index.search_molecule(
                spectra[molecule_id],
                top_n=args.top_analogs,
                mass_window_da=args.mass_window_da,
                mz_tolerance_da=args.mz_tolerance_da,
                exclude_inchikey14={molecule_id},
                require_same_polarity=args.require_same_polarity,
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
            valid_analog_counts.append(len(analog_fingerprints))
            group = group.copy()
            keys = group["inchikey14"].astype(str).tolist()
            absent = [key for key in keys if key not in fingerprint_lookup]
            if absent:
                raise ValueError(
                    f"{len(absent)} candidates for {molecule_id} are absent from fingerprint index"
                )
            indices = np.asarray([fingerprint_lookup[key] for key in keys], dtype=np.int64)
            candidate_fingerprints = fingerprint_index.unpack(indices)
            analog_matrix = (
                np.stack(analog_fingerprints)
                if analog_fingerprints
                else np.empty((0, fingerprint_index.n_bits), dtype=np.uint8)
            )
            for name, values in score_analog_fingerprint_features(
                candidate_fingerprints,
                analog_matrix,
                analog_similarities,
            ).items():
                group[name] = values
            groups.append(group)

    scored = pd.concat(groups, ignore_index=True)
    scored = prepare_analog_ranking_features(scored)
    output = (
        scored
        if args.features_only
        else fuse_analog_ranks(
            scored,
            analog_weight=args.analog_weight,
            rrf_k=args.rrf_k,
            baseline_rank_column="rank",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    report = {
        "rows": len(output),
        "queries": int(output["molecule_id"].nunique()),
        "representatives": len(analog_index),
        "representative_structures": analog_index.structure_count,
        "representative_peaks": analog_index.peak_count,
        "top_analogs": args.top_analogs,
        "max_query_spectra": args.max_query_spectra,
        "mass_window_da": args.mass_window_da,
        "require_same_polarity": args.require_same_polarity,
        "analog_weight": args.analog_weight,
        "rrf_k": args.rrf_k,
        "features_only": args.features_only,
        "median_valid_analogs": float(np.median(valid_analog_counts)),
        "runtime_seconds": time.perf_counter() - started,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
