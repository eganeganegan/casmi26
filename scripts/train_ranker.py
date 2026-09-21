#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi.ranking import RANKING_FEATURES, feature_matrix
from casmi.utils.config import load_config
from casmi.utils.random import seed_everything
from casmi.validation import evaluate_candidate_recall, evaluate_mrr


def _sorted_group(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    ordered = frame.sort_values(["molecule_id", "mass_rank"], kind="stable").reset_index(drop=True)
    groups = ordered.groupby("molecule_id", sort=False).size().astype(int).tolist()
    return ordered, groups


def _ranked_predictions(
    frame: pd.DataFrame,
    score_column: str,
    *,
    ascending: bool,
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        ordered = group.sort_values(
            [score_column, "mass_rank"],
            ascending=[ascending, True],
            kind="stable",
        )
        predictions[str(molecule_id)] = ordered["inchikey14"].astype(str).tolist()
    return predictions


def _rrf_predictions(
    frame: pd.DataFrame,
    *,
    mass_weight: float,
    rrf_k: float,
) -> dict[str, list[str]]:
    values = frame.copy()
    values["rrf_score"] = (
        mass_weight / (rrf_k + values["mass_rank"].astype(float))
        + (1.0 - mass_weight) / (rrf_k + values["fingerprint_rank"].astype(float))
    )
    return _ranked_predictions(values, "rrf_score", ascending=False)


def _truth(feature_path: Path, frame: pd.DataFrame) -> dict[str, str]:
    manifest = feature_path.parent / "query_ids.csv"
    if manifest.exists():
        ids = pd.read_csv(manifest)["molecule_id"].astype(str)
    else:
        ids = frame["molecule_id"].astype(str).drop_duplicates()
    return {molecule_id: molecule_id for molecule_id in ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-safe LightGBM candidate ranker")
    parser.add_argument("--config", type=Path, default=Path("configs/ranker.yaml"))
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--report-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-feature",
        action="append",
        default=[],
        help="Additional numeric feature column; may be supplied repeatedly",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    started = time.perf_counter()
    train_frame = pd.read_parquet(args.train_features)
    report_frame = pd.read_parquet(args.report_features)
    feature_columns = tuple(RANKING_FEATURES) + tuple(dict.fromkeys(args.extra_feature))
    missing = set(feature_columns) - set(train_frame.columns)
    if missing:
        raise ValueError(f"Training frame is missing features: {sorted(missing)}")
    if set(feature_columns) - set(report_frame.columns):
        raise ValueError("Report frame does not contain the training feature schema")

    # LambdaRank has no learning signal in groups without a positive candidate.
    positive_by_query = train_frame.groupby("molecule_id")["target"].max()
    positive_queries = sorted(positive_by_query[positive_by_query == 1].index.astype(str))
    if len(positive_queries) < 2:
        raise ValueError("At least two positive training queries are required")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(positive_queries)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * float(config.get("validation_fraction", 0.2)))))
    validation_ids = set(shuffled[:validation_count].tolist())
    fitting_ids = set(shuffled[validation_count:].tolist())
    fitting, fitting_groups = _sorted_group(
        train_frame[train_frame["molecule_id"].astype(str).isin(fitting_ids)]
    )
    validation, validation_groups = _sorted_group(
        train_frame[train_frame["molecule_id"].astype(str).isin(validation_ids)]
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=int(config.get("n_estimators", 1000)),
        learning_rate=float(config.get("learning_rate", 0.03)),
        num_leaves=int(config.get("num_leaves", 31)),
        min_child_samples=int(config.get("min_child_samples", 20)),
        subsample=float(config.get("subsample", 0.9)),
        colsample_bytree=float(config.get("colsample_bytree", 0.9)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=int(config.get("n_jobs", -1)),
        verbosity=-1,
    )
    model.fit(
        feature_matrix(fitting, feature_columns),
        fitting["target"].astype(int),
        group=fitting_groups,
        eval_X=feature_matrix(validation, feature_columns),
        eval_y=validation["target"].astype(int),
        eval_group=[validation_groups],
        eval_metric="ndcg",
        eval_at=(1, 5, 10, 25),
        callbacks=[
            lgb.early_stopping(int(config.get("early_stopping_rounds", 50)), verbose=False),
            lgb.log_evaluation(int(config.get("log_period", 50))),
        ],
    )
    report_frame = report_frame.copy()
    report_frame["ranker_score"] = model.predict(
        feature_matrix(report_frame, feature_columns), num_iteration=model.best_iteration_
    )
    truth = _truth(args.report_features, report_frame)
    prediction_sets = {
        "mass_only": _ranked_predictions(report_frame, "mass_rank", ascending=True),
        "fingerprint_only": _ranked_predictions(
            report_frame, "fingerprint_rank", ascending=True
        ),
        "rrf": _rrf_predictions(
            report_frame,
            mass_weight=float(config.get("rrf_mass_weight", 0.5)),
            rrf_k=float(config.get("rrf_k", 60.0)),
        ),
        "fragmentation_only": _ranked_predictions(
            report_frame, "fragment_intensity_fraction_mean", ascending=False
        ),
        "lightgbm_ranker": _ranked_predictions(
            report_frame, "ranker_score", ascending=False
        ),
    }
    metrics = {
        name: {
            **evaluate_mrr(truth, predictions).to_dict(),
            **evaluate_candidate_recall(truth, predictions, (25, 100, 500, 5000)),
        }
        for name, predictions in prediction_sets.items()
    }
    metrics.update(
        {
            "best_iteration": int(model.best_iteration_),
            "feature_count": len(feature_columns),
            "train_positive_queries": len(fitting_ids),
            "validation_positive_queries": len(validation_ids),
            "excluded_zero_positive_queries": int((positive_by_query == 0).sum()),
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    final_order = report_frame.sort_values(
        ["molecule_id", "ranker_score", "mass_rank"],
        ascending=[True, False, True],
        kind="stable",
    )
    final_rank = pd.Series(index=report_frame.index, dtype=np.int64)
    final_rank.loc[final_order.index] = (
        final_order.groupby("molecule_id", sort=False).cumcount().to_numpy() + 1
    )
    report_frame["final_rank"] = final_rank.astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, args.output / "config.yaml")
    model.booster_.save_model(str(args.output / "ranker.txt"))
    report_frame.sort_values(["molecule_id", "final_rank"]).to_parquet(
        args.output / "predictions.parquet", index=False
    )
    pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False).to_csv(
        args.output / "feature_importance.csv", index=False
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "feature_columns.json").write_text(
        json.dumps(list(feature_columns), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
