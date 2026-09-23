#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi.pipeline import CASMIPipeline
from casmi.retrieval import aggregate_hits
from casmi.spectra import Spectrum


def _query_from_index(index: object, spectrum_index: int) -> Spectrum:
    start = int(index.peak_offsets[spectrum_index])
    end = int(index.peak_offsets[spectrum_index + 1])
    polarity = int(index.polarities[spectrum_index])
    structure_code = int(index.structure_codes[spectrum_index])
    collision_energy = float(index.collision_energies[spectrum_index])
    return Spectrum(
        molecule_id=index.structure_keys[structure_code],
        spectrum_id=index.spectrum_ids[spectrum_index],
        precursor_mz=float(index.precursor_mz[spectrum_index]),
        adduct=index.adduct_values[int(index.adduct_codes[spectrum_index])],
        collision_energy=(collision_energy if np.isfinite(collision_energy) else None),
        mz=index.peak_mz[start:end],
        intensity=index.peak_intensity[start:end],
        ionization_mode=(
            "positive" if polarity > 0 else "negative" if polarity < 0 else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate leave-one-spectrum-out Class-1 spectral retrieval"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument(
        "--query-ids",
        type=Path,
        help="Optional CSV containing molecule_id values to evaluate in listed order",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--ingest-lib")
    parser.add_argument("--mass-tolerance-ppm", type=float, default=8.5)
    parser.add_argument("--mz-tolerance-da", type=float, default=0.02)
    parser.add_argument("--top-spectra", type=int, default=500)
    args = parser.parse_args()
    if args.queries < 0:
        parser.error("--queries must be non-negative")
    if (args.train is None) != (args.ingest_lib is None):
        parser.error("--train and --ingest-lib must be supplied together")

    pipeline = CASMIPipeline.load(args.model)
    if pipeline.index is None:
        raise ValueError("Serialized pipeline has no spectral index")
    index = pipeline.index
    counts = np.bincount(
        index.structure_codes.astype(np.int64), minlength=len(index.structure_keys)
    )
    if args.train is not None:
        ingest = pq.read_table(args.train, columns=["ingest_lib"])["ingest_lib"]
        if len(ingest) != len(index):
            raise ValueError(
                "Training parquet row count does not match the serialized spectral index"
            )
        source_indices = np.flatnonzero(
            pc.equal(ingest, args.ingest_lib).to_numpy(zero_copy_only=False)
        )
        source_codes = np.unique(index.structure_codes[source_indices].astype(np.int64))
        eligible = source_codes[counts[source_codes] >= 2]
    else:
        source_indices = np.arange(len(index), dtype=np.int64)
        eligible = np.flatnonzero(counts >= 2)
    if not len(eligible):
        raise ValueError("No structures have at least two library spectra")
    if args.query_ids:
        requested = pd.read_csv(args.query_ids)["molecule_id"].astype(str).tolist()
        code_by_key = {
            str(index.structure_keys[code]): code
            for code in eligible
        }
        selected_codes = np.asarray(
            [code_by_key[key] for key in requested if key in code_by_key],
            dtype=np.int64,
        )
        if args.queries:
            selected_codes = selected_codes[: args.queries]
    else:
        if args.queries == 0:
            parser.error("--queries=0 requires --query-ids")
        rng = np.random.default_rng(args.seed)
        selected_codes = rng.choice(
            eligible, size=min(args.queries, len(eligible)), replace=False
        )
    pending = set(map(int, selected_codes))
    selected_indices: list[int] = []
    for raw_index in source_indices:
        spectrum_index = int(raw_index)
        code = int(index.structure_codes[spectrum_index])
        if code in pending:
            selected_indices.append(spectrum_index)
            pending.remove(code)
            if not pending:
                break

    rows: list[dict[str, object]] = []
    for query_number, spectrum_index in enumerate(selected_indices, start=1):
        query = _query_from_index(index, spectrum_index)
        truth_key = str(query.molecule_id)
        hits = index.search(
            query,
            top_n=args.top_spectra,
            mz_tolerance=args.mz_tolerance_da,
            mass_tolerance_ppm=args.mass_tolerance_ppm,
            exclude_library_spectrum_ids={query.spectrum_id},
        )
        evidence = aggregate_hits(hits, method="top_k_mean")
        truth_rank = next(
            (rank for rank, candidate in enumerate(evidence, start=1) if candidate.inchikey14 == truth_key),
            None,
        )
        top = evidence[0] if evidence else None
        rows.append(
            {
                "query": query_number,
                "truth_inchikey14": truth_key,
                "truth_rank": truth_rank,
                "top_correct": bool(top is not None and top.inchikey14 == truth_key),
                "top_inchikey14": top.inchikey14 if top is not None else None,
                "top_score": top.score if top is not None else 0.0,
                "top_max_cosine": top.max_cosine if top is not None else 0.0,
                "top_explained_intensity": (
                    top.explained_query_intensity if top is not None else 0.0
                ),
                "top_matched_peaks": top.matched_peaks if top is not None else 0,
            }
        )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    ranks = pd.to_numeric(frame["truth_rank"], errors="coerce")
    report: dict[str, object] = {
        "queries": len(frame),
        "requested_query_ids": str(args.query_ids) if args.query_ids else None,
        "ingest_lib": args.ingest_lib,
        "mrr_at_25": float(((1.0 / ranks).where(ranks.le(25), 0.0)).fillna(0.0).mean()),
        "hits_at_1": float(frame["top_correct"].mean()),
        "hits_at_25": float(ranks.le(25).fillna(False).mean()),
        "thresholds": [],
    }
    for threshold in (0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        gated = (
            frame["top_max_cosine"].ge(threshold)
            & frame["top_explained_intensity"].ge(0.70)
            & frame["top_matched_peaks"].ge(6)
        )
        report["thresholds"].append(
            {
                "min_cosine": threshold,
                "coverage": float(gated.mean()),
                "precision": float(frame.loc[gated, "top_correct"].mean()) if gated.any() else None,
                "correct_queries": int((gated & frame["top_correct"]).sum()),
                "gated_queries": int(gated.sum()),
            }
        )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
