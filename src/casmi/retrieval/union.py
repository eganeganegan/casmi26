from __future__ import annotations

import pandas as pd


def prune_wide_fallback_candidates(
    candidates: pd.DataFrame,
    *,
    max_wide_fingerprint_rank: int,
) -> pd.DataFrame:
    """Keep the precise-mass pool plus a bounded wide-window fingerprint fallback."""
    if max_wide_fingerprint_rank < 0:
        raise ValueError("max_wide_fingerprint_rank must be non-negative")
    required = {
        "molecule_id",
        "inchikey14",
        "within_20ppm",
        "wide_mass_rank",
        "wide_fingerprint_rank",
        "absolute_mass_error_ppm",
        "predicted_fingerprint_score",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate rows are missing columns: {sorted(missing)}")
    if candidates.duplicated(["molecule_id", "inchikey14"]).any():
        raise ValueError("Candidate rows repeat a query/InChIKey14 pair")
    keep = candidates["within_20ppm"].eq(1) | candidates["wide_fingerprint_rank"].le(
        max_wide_fingerprint_rank
    )
    output = candidates.loc[keep].copy()
    output["mass_rank"] = (
        output.groupby("molecule_id", sort=False)["wide_mass_rank"]
        .rank(method="first", ascending=True)
        .astype("int32")
    )
    output["rank"] = (
        output.groupby("molecule_id", sort=False)["wide_fingerprint_rank"]
        .rank(method="first", ascending=True)
        .astype("int32")
    )
    output["candidate_count"] = (
        output.groupby("molecule_id", sort=False)["inchikey14"]
        .transform("size")
        .astype("int32")
    )
    return output.sort_values(["molecule_id", "rank"], kind="stable").reset_index(drop=True)
