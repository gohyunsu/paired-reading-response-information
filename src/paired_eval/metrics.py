from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def strict_auc(labels, scores, sample_weight=None) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must have equal length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUROC requires both binary outcomes")
    return float(roc_auc_score(labels, scores, sample_weight=sample_weight))


def within_question_auc(metadata: pd.DataFrame, scores) -> float:
    frame = metadata[["item_key", "label"]].copy()
    frame["score"] = np.asarray(scores, dtype=float)
    favorable = 0.0
    total = 0
    for _, group in frame.groupby("item_key", sort=False):
        positive = group.loc[group.label.eq(1), "score"].to_numpy()
        negative = group.loc[group.label.eq(0), "score"].to_numpy()
        if not len(positive) or not len(negative):
            continue
        difference = positive[:, None] - negative[None, :]
        favorable += float((difference > 0).sum() + 0.5 * (difference == 0).sum())
        total += difference.size
    return favorable / total if total else 0.5


def evaluate(metadata: pd.DataFrame, scores, regime) -> dict[str, float]:
    labels = metadata.label.to_numpy(int)
    prediction = np.asarray(scores, dtype=float)
    regime = np.asarray(regime)
    result = {
        name: strict_auc(labels[regime == name], prediction[regime == name])
        for name in ("UT", "UR", "UB")
    }
    result.update(
        macro=float(np.mean([result["UT"], result["UR"], result["UB"]])),
        pooled=strict_auc(labels, prediction),
        brier=float(brier_score_loss(labels, prediction)),
        log_loss=float(log_loss(labels, np.clip(prediction, 1e-7, 1 - 1e-7))),
        within_question=float(within_question_auc(metadata, prediction)),
    )
    return result


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        raise ValueError("At least one metric record is required")
    if any(set(record) != set(records[0]) for record in records):
        raise ValueError("Metric records have different fields")
    return {
        key: float(np.mean([record[key] for record in records])) for key in records[0]
    }


def rank_record(record: dict) -> tuple[float, float, str]:
    """Higher macro AUROC, lower Brier score, then stable key."""
    return (-record["mean"]["macro"], record["mean"]["brier"], record["key"])
