from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .offsets import PENALTIES, fit_scalar_offset
from .splits import cartesian_splits
from .utils import read_json, sha256, write_json


MINIMUM_HISTORY = 10


def _load_base_predictions(design: Path, primary: Path) -> pd.DataFrame:
    assigned = pd.read_csv(design / "OneStop_RC" / "outer_assignment.csv")
    frames = []
    for split in cartesian_splits(assigned):
        if not np.any(split.regime == "UB"):
            continue
        name = f"OneStop_RC_r{split.reader_fold}_m{split.material_fold}"
        folder = primary / "OneStop_RC" / name
        completion = read_json(folder / "complete.json")
        prediction_path = folder / "predictions_unscored.npz"
        if sha256(prediction_path) != completion["prediction_sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {name}")
        target = assigned.iloc[split.test].reset_index(drop=True)
        with np.load(prediction_path) as stored:
            np.testing.assert_array_equal(stored["ids"], target.unique_trial_id.astype(str))
            np.testing.assert_array_equal(stored["regime"], split.regime)
            mask = split.regime == "UB"
            part = target.loc[
                mask, ["unique_trial_id", "participant_id", "label"]
            ].copy()
            part["base"] = stored["base"][mask]
        frames.append(part)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != len(assigned) or result.unique_trial_id.duplicated().any():
        raise ValueError("Expected exactly one UB base prediction per OneStop trial")
    return result


def _load_order(path: Path, universe: set[str]) -> pd.DataFrame:
    order = pd.read_csv(path)
    required = {"unique_trial_id", "participant_id", "trial_index"}
    missing = required - set(order)
    if missing:
        raise ValueError(f"Trial-order file is missing columns: {sorted(missing)}")
    order = order.loc[
        order.unique_trial_id.astype(str).isin(universe), list(required)
    ].copy()
    order["unique_trial_id"] = order.unique_trial_id.astype(str)
    if set(order.unique_trial_id) != universe:
        raise ValueError("Trial-order file does not cover every evaluated trial")
    if order.unique_trial_id.duplicated().any():
        raise ValueError("Trial-order file has duplicate trial identifiers")
    if order.duplicated(["participant_id", "trial_index"]).any():
        raise ValueError("Trial order is tied within a participant")
    return order


def _point_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    labels = frame.label.to_numpy(int)
    result = {
        "pooled_auroc": float(roc_auc_score(labels, prediction)),
        "brier": float(brier_score_loss(labels, prediction)),
        "log_loss": float(log_loss(labels, np.clip(prediction, 1e-7, 1 - 1e-7))),
    }
    reader_auc = []
    for _, group in frame.assign(prediction=prediction).groupby("participant_id"):
        if group.label.nunique() == 2:
            reader_auc.append(roc_auc_score(group.label, group.prediction))
    result["mean_within_reader_auroc"] = float(np.mean(reader_auc))
    result["eligible_readers"] = len(reader_auc)
    return result


def _weighted_pooled(labels, prediction, weights) -> dict:
    total = float(weights.sum())
    clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
    return {
        "pooled_auroc": float(
            roc_auc_score(labels, prediction, sample_weight=weights)
        ),
        "brier": float(np.sum(weights * (labels - prediction) ** 2) / total),
        "log_loss": float(
            -np.sum(
                weights
                * (labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
            )
            / total
        ),
    }


def _interval(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "percentile_95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def _bootstrap(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> dict:
    reader_ids = np.array(sorted(frame.participant_id.astype(str).unique()))
    reader_code = pd.Categorical(
        frame.participant_id.astype(str), categories=reader_ids
    ).codes
    labels = frame.label.to_numpy(int)
    per_reader = {name: np.full(len(reader_ids), np.nan) for name in predictions}
    for reader, group in frame.assign(row=np.arange(len(frame))).groupby("participant_id"):
        position = int(np.flatnonzero(reader_ids == str(reader))[0])
        if group.label.nunique() != 2:
            continue
        rows = group.row.to_numpy(int)
        for name, prediction in predictions.items():
            per_reader[name][position] = roc_auc_score(labels[rows], prediction[rows])
    eligible = np.isfinite(per_reader["base"])
    if not all(np.array_equal(np.isfinite(value), eligible) for value in per_reader.values()):
        raise ValueError("Eligible-reader set differs across predictions")
    contrasts = {
        name: {
            metric: np.empty(repetitions, dtype=float)
            for metric in ("pooled_auroc", "brier", "log_loss", "mean_within_reader_auroc")
        }
        for name in predictions
        if name != "base"
    }
    rng = np.random.default_rng(seed)
    for replicate in range(repetitions):
        multiplicity = rng.multinomial(
            len(reader_ids), np.full(len(reader_ids), 1 / len(reader_ids))
        )
        weights = multiplicity[reader_code].astype(float)
        reference = _weighted_pooled(labels, predictions["base"], weights)
        reference["mean_within_reader_auroc"] = float(
            np.average(per_reader["base"][eligible], weights=multiplicity[eligible])
        )
        for name, prediction in predictions.items():
            if name == "base":
                continue
            score = _weighted_pooled(labels, prediction, weights)
            score["mean_within_reader_auroc"] = float(
                np.average(per_reader[name][eligible], weights=multiplicity[eligible])
            )
            for metric in contrasts[name]:
                contrasts[name][metric][replicate] = score[metric] - reference[metric]
    return {
        "attempted": repetitions,
        "valid": repetitions,
        "seed": seed,
        "eligible_readers": int(eligible.sum()),
        "paired_differences_vs_base": {
            name: {metric: _interval(values) for metric, values in metrics.items()}
            for name, metrics in contrasts.items()
        },
    }


def _plot(report: dict, output: Path) -> None:
    penalties = [str(value) for value in PENALTIES]
    bootstrap = report["bootstrap"]["paired_differences_vs_base"]
    pooled = [bootstrap[f"penalty_{value}"]["pooled_auroc"] for value in PENALTIES]
    within = [
        bootstrap[f"penalty_{value}"]["mean_within_reader_auroc"]
        for value in PENALTIES
    ]
    figure, axis = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(penalties))
    for records, label, color, marker in (
        (pooled, "Pooled", "#2c5f8a", "o"),
        (within, "Within reader", "#777777", "s"),
    ):
        point = np.array([record["mean"] for record in records]) * 100
        bounds = np.array([record["percentile_95"] for record in records]) * 100
        axis.errorbar(
            x,
            point,
            yerr=np.vstack([point - bounds[:, 0], bounds[:, 1] - point]),
            label=label,
            color=color,
            marker=marker,
            linewidth=1.1,
            capsize=2.5,
        )
    axis.axhline(0, color="#aaaaaa", linewidth=0.8)
    axis.set_xticks(x, penalties)
    axis.set_xlabel("Offset penalty")
    axis.set_ylabel("AUROC gain (percentage points)")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def run_prequential(
    design: Path,
    primary: Path,
    trial_order: Path,
    output: Path,
    *,
    minimum_history: int = MINIMUM_HISTORY,
    bootstrap_repetitions: int = 5000,
    seed: int = 20260926,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frame = _load_base_predictions(design, primary)
    order = _load_order(trial_order, set(frame.unique_trial_id.astype(str)))
    frame = frame.merge(
        order[["unique_trial_id", "trial_index"]],
        on="unique_trial_id",
        validate="one_to_one",
    )
    frame = frame.sort_values(["participant_id", "trial_index"]).reset_index(drop=True)
    frame["history_count"] = frame.groupby("participant_id").cumcount()
    predictions = {
        penalty: np.empty(len(frame), dtype=float) for penalty in PENALTIES
    }
    for _, group in frame.groupby("participant_id", sort=False):
        rows = group.index.to_numpy()
        base = group.base.to_numpy(float)
        labels = group.label.to_numpy(int)
        for position, row in enumerate(rows):
            for penalty in PENALTIES:
                offset = fit_scalar_offset(base[:position], labels[:position], penalty)
                predictions[penalty][row] = expit(
                    logit(np.clip(base[position], 1e-5, 1 - 1e-5)) + offset
                )
    prediction_path = output / "predictions_unscored.npz"
    np.savez_compressed(
        prediction_path,
        ids=frame.unique_trial_id.to_numpy(dtype=str),
        history_count=frame.history_count.to_numpy(int),
        base=frame.base.to_numpy(float),
        **{f"penalty_{value}": prediction for value, prediction in predictions.items()},
    )
    mask = frame.history_count.to_numpy() >= minimum_history
    retained = frame.loc[mask].reset_index(drop=True)
    retained_predictions = {"base": frame.base.to_numpy()[mask]}
    retained_predictions.update(
        {f"penalty_{value}": prediction[mask] for value, prediction in predictions.items()}
    )
    point = {
        name: _point_metrics(retained, prediction)
        for name, prediction in retained_predictions.items()
    }
    bootstrap = _bootstrap(
        retained, retained_predictions, bootstrap_repetitions, seed
    )
    report = {
        "minimum_history": minimum_history,
        "rows": len(retained),
        "readers": int(retained.participant_id.nunique()),
        "penalties": list(PENALTIES),
        "point": point,
        "bootstrap": bootstrap,
        "prediction_sha256": sha256(prediction_path),
    }
    write_json(output / "evaluation.json", report)
    _plot(report, output / "prequential_auroc.pdf")
    return report
