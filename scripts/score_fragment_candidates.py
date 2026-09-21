#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase
from tqdm import tqdm

from casmi.data import iter_parquet_spectra
from casmi.pipeline import CASMIPipeline
from casmi.ranking.fragmentation import (
    FRAGMENT_FEATURES,
    FragmentationConfig,
    score_candidate_fragments,
)
from casmi.spectra import Spectrum, clean_spectrum
from casmi.utils.config import load_config


def _query_spectra(
    index: object, query_ids: set[str], max_spectra: int
) -> dict[str, list[Spectrum]]:
    code_by_key = {
        index.structure_keys[code]: code
        for code in range(len(index.structure_keys))
        if index.structure_keys[code] in query_ids
    }
    key_by_code = {code: key for key, code in code_by_key.items()}
    spectra: dict[str, list[Spectrum]] = {key: [] for key in query_ids}
    for spectrum_index, raw_code in enumerate(index.structure_codes):
        code = int(raw_code)
        key = key_by_code.get(code)
        if key is None or len(spectra[key]) >= max_spectra:
            continue
        start = int(index.peak_offsets[spectrum_index])
        end = int(index.peak_offsets[spectrum_index + 1])
        polarity = int(index.polarities[spectrum_index])
        spectra[key].append(
            Spectrum(
                molecule_id=key,
                spectrum_id=index.spectrum_ids[spectrum_index],
                precursor_mz=float(index.precursor_mz[spectrum_index]),
                adduct=index.adduct_values[int(index.adduct_codes[spectrum_index])],
                collision_energy=float(index.collision_energies[spectrum_index]),
                mz=index.peak_mz[start:end],
                intensity=index.peak_intensity[start:end],
                ionization_mode=(
                    "positive" if polarity > 0 else "negative" if polarity < 0 else None
                ),
            )
        )
    return spectra


def _parquet_query_spectra(
    path: Path, query_ids: set[str], max_spectra: int
) -> dict[str, list[Spectrum]]:
    spectra: dict[str, list[Spectrum]] = {key: [] for key in query_ids}
    for spectrum in iter_parquet_spectra(path, molecule_ids=query_ids):
        values = spectra.get(str(spectrum.molecule_id))
        if values is None or len(values) >= max_spectra:
            continue
        mz, intensity = clean_spectrum(
            spectrum.mz, spectrum.intensity, precursor_mz=spectrum.precursor_mz
        )
        spectrum.mz = mz
        spectrum.intensity = intensity
        # Structure annotations are never needed by fragmentation inference.
        spectrum.smiles = None
        spectrum.canonical_smiles = None
        spectrum.inchikey14 = None
        spectrum.exact_mass = None
        spectrum.molecular_formula = None
        values.append(spectrum)
    return spectra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add bounded MetFrag-lite features to candidate rows"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/fragmentation.yaml"))
    parser.add_argument("--candidates", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spectral-index", type=Path)
    source.add_argument("--spectra", type=Path, help="Raw train/test parquet for query spectra")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Prior candidate parquet whose already-scored query/candidate pairs can be reused",
    )
    parser.add_argument("--top-candidates", type=int)
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--max-bonds", type=int)
    parser.add_argument("--mz-tolerance-da", type=float)
    parser.add_argument("--top-intensity-peaks", type=int)
    parser.add_argument("--min-fragment-mass", type=float)
    args = parser.parse_args()
    raw_config = load_config(args.config)
    top_candidates = int(
        args.top_candidates
        if args.top_candidates is not None
        else raw_config.get("top_candidates", 50)
    )
    max_spectra = int(
        args.max_spectra if args.max_spectra is not None else raw_config.get("max_spectra", 8)
    )
    max_bonds = int(
        args.max_bonds if args.max_bonds is not None else raw_config.get("max_bonds", 32)
    )
    mz_tolerance_da = float(
        args.mz_tolerance_da
        if args.mz_tolerance_da is not None
        else raw_config.get("mz_tolerance_da", 0.02)
    )
    top_intensity_peaks = int(
        args.top_intensity_peaks
        if args.top_intensity_peaks is not None
        else raw_config.get("top_intensity_peaks", 10)
    )
    min_fragment_mass = float(
        args.min_fragment_mass
        if args.min_fragment_mass is not None
        else raw_config.get("min_fragment_mass", 15.0)
    )
    if top_candidates <= 0:
        parser.error("--top-candidates must be positive")
    started = time.perf_counter()
    candidates = pd.read_parquet(args.candidates)
    required = {"molecule_id", "smiles", "mass_rank", "rank"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate rows are missing columns: {sorted(missing)}")
    if candidates.duplicated(["molecule_id", "inchikey14"]).any():
        raise ValueError("Candidate rows repeat a query/InChIKey14 pair")
    query_ids = set(candidates["molecule_id"].astype(str))
    if args.spectral_index is not None:
        pipeline = CASMIPipeline.load(args.spectral_index)
        if pipeline.index is None:
            raise ValueError("Serialized pipeline has no spectral index")
        spectra = _query_spectra(pipeline.index, query_ids, max_spectra)
    else:
        spectra = _parquet_query_spectra(args.spectra, query_ids, max_spectra)
    missing_spectra = sorted(key for key, values in spectra.items() if not values)
    if missing_spectra:
        raise ValueError(f"Queries missing from spectral index: {missing_spectra[:5]}")

    output = candidates.copy()
    for feature in FRAGMENT_FEATURES:
        output[feature] = np.float32(0.0)
    priority = output[["mass_rank", "rank"]].min(axis=1)
    selected = (
        output.assign(_fragment_priority=priority)
        .sort_values(["molecule_id", "_fragment_priority", "mass_rank"], kind="stable")
        .groupby("molecule_id", sort=False)
        .head(top_candidates)
    )
    selected_total = len(selected)
    cache_reused = 0
    if args.feature_cache is not None:
        cache_columns = ["molecule_id", "inchikey14", *FRAGMENT_FEATURES]
        cache = pd.read_parquet(args.feature_cache, columns=cache_columns)
        cache = cache.loc[cache["fragment_scored"].eq(1)]
        if cache.duplicated(["molecule_id", "inchikey14"]).any():
            raise ValueError("Fragment feature cache repeats a query/InChIKey14 pair")
        cached = (
            selected[["molecule_id", "inchikey14"]]
            .assign(_row_index=selected.index)
            .merge(cache, on=["molecule_id", "inchikey14"], how="inner", validate="one_to_one")
        )
        cache_reused = len(cached)
        for feature in FRAGMENT_FEATURES:
            output.loc[cached["_row_index"], feature] = cached[feature].to_numpy(
                dtype=np.float32
            )
    selected = selected.loc[output.loc[selected.index, "fragment_scored"].eq(0).to_numpy()]
    config = FragmentationConfig(
        mz_tolerance_da=mz_tolerance_da,
        max_bonds=max_bonds,
        max_spectra=max_spectra,
        top_intensity_peaks=top_intensity_peaks,
        min_fragment_mass=min_fragment_mass,
    )
    invalid = 0
    with rdBase.BlockLogs():
        for row in tqdm(selected.itertuples(), total=len(selected), desc="fragment scoring"):
            try:
                values = score_candidate_fragments(
                    str(row.smiles), spectra[str(row.molecule_id)], config
                )
            except (ValueError, RuntimeError):
                invalid += 1
                continue
            for feature, value in values.items():
                output.at[row.Index, feature] = np.float32(value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    query_manifest = args.candidates.parent / "query_ids.csv"
    destination_manifest = args.output.parent / "query_ids.csv"
    if query_manifest.exists() and query_manifest.resolve() != destination_manifest.resolve():
        shutil.copyfile(query_manifest, destination_manifest)
    report = {
        "rows": len(output),
        "queries": int(output["molecule_id"].nunique()),
        "selected_rows": selected_total,
        "newly_scored_rows": len(selected),
        "scored_rows": int(output["fragment_scored"].sum()),
        "cache_reused_rows": cache_reused,
        "invalid_candidates": invalid,
        "top_candidates": top_candidates,
        "max_spectra": max_spectra,
        "max_bonds": max_bonds,
        "mz_tolerance_da": mz_tolerance_da,
        "top_intensity_peaks": top_intensity_peaks,
        "min_fragment_mass": min_fragment_mass,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output.parent / f"{args.output.stem}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
