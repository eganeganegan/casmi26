from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import rdBase

from casmi.chemistry import basic_descriptors
from casmi.ranking.fragmentation import FRAGMENT_FEATURES

BASE_FEATURES = (
    "mass_rank",
    "reciprocal_mass_rank",
    "mass_rank_percentile",
    "mass_error_ppm",
    "absolute_mass_error_ppm",
    "log_absolute_mass_error_ppm",
    "mass_score_20ppm",
    "within_20ppm",
    "within_50ppm",
    "within_1000ppm",
    "predicted_fingerprint_score",
    "fingerprint_rank",
    "reciprocal_fingerprint_rank",
    "fingerprint_rank_percentile",
    "fingerprint_score_z",
    "wide_mass_rank",
    "wide_fingerprint_rank",
    "wide_mass_rank_percentile",
    "wide_fingerprint_rank_percentile",
    "wide_candidate_count",
    "candidate_exact_mass",
    "neutral_mass",
    "candidate_count",
    "query_spectrum_count",
    "query_adduct_count",
    "query_positive_fraction",
    "query_mean_collision_energy",
    "source_coconut",
    "source_pubchemlite",
    "source_chembl",
    "has_formula",
)

DESCRIPTOR_FEATURES = (
    "candidate_heteroatom_count",
    "candidate_ring_count",
    "candidate_aromatic_fraction",
    "candidate_logp",
)

RANKING_FEATURES = BASE_FEATURES + DESCRIPTOR_FEATURES + FRAGMENT_FEATURES


def _descriptor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    structures = frame[["inchikey14", "smiles"]].drop_duplicates("inchikey14")
    rows: list[dict[str, object]] = []
    with rdBase.BlockLogs():
        for row in structures.itertuples(index=False):
            try:
                values = basic_descriptors(str(row.smiles))
            except (ValueError, RuntimeError):
                values = {
                    "heteroatom_count": np.nan,
                    "ring_count": np.nan,
                    "aromatic_fraction": np.nan,
                    "logp": np.nan,
                }
            rows.append(
                {
                    "inchikey14": str(row.inchikey14),
                    "candidate_heteroatom_count": values["heteroatom_count"],
                    "candidate_ring_count": values["ring_count"],
                    "candidate_aromatic_fraction": values["aromatic_fraction"],
                    "candidate_logp": values["logp"],
                }
            )
    return pd.DataFrame(rows)


def prepare_ranking_features(
    candidates: pd.DataFrame,
    *,
    add_descriptors: bool = True,
    descriptor_cache: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create identity-free numeric features from candidate-generation evidence."""
    required = {
        "molecule_id",
        "inchikey14",
        "smiles",
        "source",
        "formula",
        "target",
        "rank",
        "mass_rank",
        "mass_error_ppm",
        "absolute_mass_error_ppm",
        "mass_score_20ppm",
        "within_20ppm",
        "within_50ppm",
        "within_1000ppm",
        "predicted_fingerprint_score",
        "candidate_exact_mass",
        "neutral_mass",
        "candidate_count",
        "query_spectrum_count",
        "query_adduct_count",
        "query_positive_fraction",
        "query_mean_collision_energy",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate rows are missing columns: {sorted(missing)}")
    frame = candidates.copy()
    if frame.duplicated(["molecule_id", "inchikey14"]).any():
        raise ValueError("Candidate rows repeat a query/InChIKey14 pair")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("target must contain only 0/1 values")
    positives = frame.groupby("molecule_id")["target"].sum()
    if (positives > 1).any():
        raise ValueError("A query contains multiple positive InChIKey14 candidates")

    frame["fingerprint_rank"] = frame["rank"].astype(np.float32)
    if "wide_mass_rank" not in frame:
        frame["wide_mass_rank"] = frame["mass_rank"]
    if "wide_fingerprint_rank" not in frame:
        frame["wide_fingerprint_rank"] = frame["fingerprint_rank"]
    if "wide_candidate_count" not in frame:
        frame["wide_candidate_count"] = frame["candidate_count"]
    frame["reciprocal_mass_rank"] = 1.0 / frame["mass_rank"].astype(float)
    frame["reciprocal_fingerprint_rank"] = 1.0 / frame["fingerprint_rank"].astype(float)
    denominator = frame["candidate_count"].clip(lower=1).astype(float)
    frame["mass_rank_percentile"] = frame["mass_rank"].astype(float) / denominator
    frame["fingerprint_rank_percentile"] = frame["fingerprint_rank"].astype(float) / denominator
    wide_denominator = frame["wide_candidate_count"].clip(lower=1).astype(float)
    frame["wide_mass_rank_percentile"] = frame["wide_mass_rank"].astype(float) / wide_denominator
    frame["wide_fingerprint_rank_percentile"] = (
        frame["wide_fingerprint_rank"].astype(float) / wide_denominator
    )
    frame["log_absolute_mass_error_ppm"] = np.log1p(
        frame["absolute_mass_error_ppm"].astype(float)
    )
    grouped_scores = frame.groupby("molecule_id")["predicted_fingerprint_score"]
    score_mean = grouped_scores.transform("mean")
    score_std = grouped_scores.transform("std").fillna(0.0)
    frame["fingerprint_score_z"] = np.divide(
        frame["predicted_fingerprint_score"] - score_mean,
        score_std,
        out=np.zeros(len(frame), dtype=np.float64),
        where=score_std.to_numpy() > 0,
    )
    source = frame["source"].fillna("").astype(str).str.lower()
    frame["source_coconut"] = source.str.contains("coconut").astype(np.uint8)
    frame["source_pubchemlite"] = source.str.contains("pubchem").astype(np.uint8)
    frame["source_chembl"] = source.str.contains("chembl").astype(np.uint8)
    frame["has_formula"] = frame["formula"].notna().astype(np.uint8)
    if add_descriptors:
        if not set(DESCRIPTOR_FEATURES) <= set(frame.columns):
            descriptors = (
                descriptor_cache if descriptor_cache is not None else _descriptor_frame(frame)
            )
            descriptor_missing = {"inchikey14", *DESCRIPTOR_FEATURES} - set(
                descriptors.columns
            )
            if descriptor_missing:
                raise ValueError(
                    f"Descriptor cache is missing columns: {sorted(descriptor_missing)}"
                )
            frame = frame.merge(
                descriptors[["inchikey14", *DESCRIPTOR_FEATURES]],
                on="inchikey14",
                how="left",
                validate="many_to_one",
            )
    else:
        for column in DESCRIPTOR_FEATURES:
            frame[column] = 0.0
    # Fragmentation is optional and computed only for a bounded candidate subset.
    for column in FRAGMENT_FEATURES:
        if column not in frame:
            frame[column] = 0.0
    for column in RANKING_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
        frame[column] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def feature_matrix(frame: pd.DataFrame, columns: Sequence[str] = RANKING_FEATURES) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Ranking frame is missing features: {sorted(missing)}")
    return frame.loc[:, list(columns)]
