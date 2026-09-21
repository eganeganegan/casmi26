from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from rdkit import rdBase

from casmi.chemistry import deduplicate_structures_by_inchikey14, sanitize_smiles


def validate_submission(frame: pd.DataFrame, expected_molecule_ids: Sequence[str] | None = None) -> None:
    if list(frame.columns) != ["molecule_id", "smiles"]:
        raise ValueError("Submission columns must be exactly ['molecule_id', 'smiles'] in that order")
    if frame.empty:
        raise ValueError("Submission is empty")
    if frame.isna().any().any():
        raise ValueError("Submission contains null values")
    if frame["molecule_id"].duplicated().any():
        raise ValueError("Submission repeats molecule_id values")
    if expected_molecule_ids is not None:
        actual = set(frame["molecule_id"].astype(str))
        expected = set(map(str, expected_molecule_ids))
        if actual != expected:
            raise ValueError(
                f"Submission molecule IDs differ: missing={sorted(expected-actual)[:5]}, "
                f"extra={sorted(actual-expected)[:5]}"
            )
    for row in frame.itertuples(index=False):
        candidates = str(row.smiles).split(";")
        if not 1 <= len(candidates) <= 25:
            raise ValueError(f"{row.molecule_id} has {len(candidates)} candidates; expected 1..25")
        invalid = [smiles for smiles in candidates if not sanitize_smiles(smiles)]
        if invalid:
            raise ValueError(f"{row.molecule_id} has invalid SMILES: {invalid[:3]}")


def make_submission(
    predictions: Mapping[str, Sequence[str]],
    output: str | Path,
    *,
    expected_molecule_ids: Sequence[str] | None = None,
    max_candidates: int = 25,
) -> pd.DataFrame:
    if not 1 <= max_candidates <= 25:
        raise ValueError("max_candidates must be between 1 and 25")
    molecule_ids = list(map(str, expected_molecule_ids)) if expected_molecule_ids is not None else sorted(predictions)
    rows: list[dict[str, str]] = []
    for molecule_id in molecule_ids:
        unique = deduplicate_structures_by_inchikey14(
            predictions.get(molecule_id, []), limit=max_candidates
        )
        if not unique:
            raise ValueError(f"No valid candidate for molecule {molecule_id!r}")
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(unique)})
    frame = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate_submission(frame, molecule_ids)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def make_submission_from_ranked_frame(
    predictions: pd.DataFrame,
    output: str | Path,
    *,
    expected_molecule_ids: Sequence[str] | None = None,
    max_candidates: int = 25,
) -> pd.DataFrame:
    """Build a submission using trusted comparison keys already produced upstream."""
    if not 1 <= max_candidates <= 25:
        raise ValueError("max_candidates must be between 1 and 25")
    required = {"molecule_id", "rank", "smiles", "inchikey14"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Ranked predictions are missing columns: {sorted(missing)}")
    ordered = predictions.sort_values(["molecule_id", "rank"], kind="stable")
    grouped = {str(key): value for key, value in ordered.groupby("molecule_id", sort=False)}
    molecule_ids = (
        list(map(str, expected_molecule_ids))
        if expected_molecule_ids is not None
        else sorted(grouped)
    )
    rows: list[dict[str, str]] = []
    with rdBase.BlockLogs():
        for molecule_id in molecule_ids:
            candidates: list[str] = []
            seen: set[str] = set()
            group = grouped.get(molecule_id)
            if group is not None:
                for row in group.itertuples(index=False):
                    key = str(row.inchikey14)
                    smiles = str(row.smiles)
                    if len(key) != 14 or key in seen or not sanitize_smiles(smiles):
                        continue
                    seen.add(key)
                    candidates.append(smiles)
                    if len(candidates) >= max_candidates:
                        break
            if not candidates:
                raise ValueError(f"No valid candidate for molecule {molecule_id!r}")
            rows.append({"molecule_id": molecule_id, "smiles": ";".join(candidates)})
    frame = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate_submission(frame, molecule_ids)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame
