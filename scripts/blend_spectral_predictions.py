#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def blend_spectral_gate(
    ranked: pd.DataFrame,
    spectral: pd.DataFrame,
    *,
    min_cosine: float = 0.90,
    min_explained_intensity: float = 0.70,
    min_matched_peaks: int = 6,
) -> pd.DataFrame:
    """Prepend one high-confidence library hit, then retain ranker order."""
    base_required = {"molecule_id", "rank", "smiles", "inchikey14"}
    spectral_required = base_required | {
        "max_cosine",
        "explained_query_intensity",
        "matched_peaks",
    }
    base_missing = base_required - set(ranked.columns)
    spectral_missing = spectral_required - set(spectral.columns)
    if base_missing:
        raise ValueError(f"Ranked predictions are missing columns: {sorted(base_missing)}")
    if spectral_missing:
        raise ValueError(f"Spectral predictions are missing columns: {sorted(spectral_missing)}")
    if not 0.0 <= min_cosine <= 1.0:
        raise ValueError("min_cosine must be between 0 and 1")
    if not 0.0 <= min_explained_intensity <= 1.0:
        raise ValueError("min_explained_intensity must be between 0 and 1")
    if min_matched_peaks < 0:
        raise ValueError("min_matched_peaks must be non-negative")

    ranked_ordered = ranked.sort_values(["molecule_id", "rank"], kind="stable")
    spectral_ordered = spectral.sort_values(["molecule_id", "rank"], kind="stable")
    spectral_top = spectral_ordered.groupby("molecule_id", sort=False).head(1).copy()
    spectral_top["spectral_gate"] = (
        spectral_top["max_cosine"].ge(min_cosine)
        & spectral_top["explained_query_intensity"].ge(min_explained_intensity)
        & spectral_top["matched_peaks"].ge(min_matched_peaks)
    )
    gated = {
        str(row.molecule_id): row
        for row in spectral_top.loc[spectral_top["spectral_gate"]].itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for molecule_id, base_group in ranked_ordered.groupby("molecule_id", sort=False):
        candidates: list[tuple[str, str, str, float]] = []
        spectral_row = gated.get(str(molecule_id))
        if spectral_row is not None:
            candidates.append(
                (
                    str(spectral_row.inchikey14),
                    str(spectral_row.smiles),
                    "spectral_gate",
                    float(spectral_row.max_cosine),
                )
            )
        candidates.extend(
            (
                str(row.inchikey14),
                str(row.smiles),
                "ranker",
                float(getattr(row, "ranker_score", 0.0)),
            )
            for row in base_group.itertuples(index=False)
        )
        seen: set[str] = set()
        for key, smiles, source, score in candidates:
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "molecule_id": str(molecule_id),
                    "rank": len(seen),
                    "smiles": smiles,
                    "inchikey14": key,
                    "blend_source": source,
                    "blend_score": score,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate high-confidence direct spectral hits ahead of learned-ranker output"
    )
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.90)
    parser.add_argument("--min-explained-intensity", type=float, default=0.70)
    parser.add_argument("--min-matched-peaks", type=int, default=6)
    args = parser.parse_args()
    output = blend_spectral_gate(
        pd.read_parquet(args.ranked),
        pd.read_parquet(args.spectral),
        min_cosine=args.min_cosine,
        min_explained_intensity=args.min_explained_intensity,
        min_matched_peaks=args.min_matched_peaks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        json.dumps(
            {
                "rows": len(output),
                "queries": int(output["molecule_id"].nunique()),
                "gated_queries": int(
                    output.loc[output["blend_source"].eq("spectral_gate"), "molecule_id"].nunique()
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
