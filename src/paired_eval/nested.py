from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import evaluate, mean_metrics, rank_record
from .models import ForestModel
from .offsets import PENALTIES, PairedOffsets, grouped_oof_predictions
from .splits import cartesian_splits, nested_splits
from .utils import object_id, read_json, sha256, write_json


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def _metric_match(actual: dict, expected: dict, context: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"Metric fields differ for {context}")
    for key in actual:
        if abs(actual[key] - expected[key]) > 1e-12:
            raise ValueError(f"Metric replay failed for {context}/{key}")


def _find_outer(assigned: pd.DataFrame, reader_fold: int, material_fold: int):
    for split in cartesian_splits(assigned):
        if (split.reader_fold, split.material_fold) == (reader_fold, material_fold):
            return split
    raise ValueError(f"Unknown outer cell r{reader_fold}/m{material_fold}")


def run_nested_cell(
    assigned: pd.DataFrame,
    views: dict[str, np.ndarray],
    dataset: str,
    reader_fold: int,
    material_fold: int,
    specifications: list[dict],
    output: Path,
    *,
    threads: int = 1,
) -> str:
    """Fit one outer cell with base-first nested selection."""
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "complete.json"
    outer = _find_outer(assigned, reader_fold, material_fold)
    seed = 20260929 + 100 * reader_fold + material_fold
    training, inner_splits = nested_splits(assigned, dataset, outer, seed, folds=3)
    local_views = {name: matrix[outer.train] for name, matrix in views.items()}
    if any(specification["view"] not in local_views for specification in specifications):
        raise ValueError("A candidate refers to an unavailable feature view")
    context = {
        "dataset": dataset,
        "reader_fold": reader_fold,
        "material_fold": material_fold,
        "seed": seed,
        "inner_cells": len(inner_splits),
        "candidate_count": len(specifications),
        "candidate_specs": specifications,
        "feature_shape": {name: list(value.shape) for name, value in views.items()},
        "feature_sha256": {name: _array_sha256(value) for name, value in views.items()},
        "training_sha256": hashlib.sha256(training.to_csv(index=False).encode()).hexdigest(),
        "selection": "base macro AUROC, Brier score, stable candidate key; penalties second",
        "penalties": list(PENALTIES),
    }
    context_path = output / "context.json"
    if context_path.exists() and read_json(context_path) != context:
        raise ValueError(f"Cell context changed: {output.name}")
    write_json(context_path, context)
    if completion.exists():
        stored = read_json(completion)
        required = (
            output / "predictions_unscored.npz",
            output / "metrics.json",
            output / "selection.json",
        )
        if not all(path.exists() for path in required):
            raise ValueError(f"Completed cell is missing artifacts: {output.name}")
        if sha256(required[0]) != stored["prediction_sha256"]:
            raise ValueError(f"Completed cell checksum mismatch: {output.name}")
        return output.name

    base_records = []
    for specification_index, specification in enumerate(specifications):
        scores = []
        feature_matrix = local_views[specification["view"]]
        cache_key = object_id(specification)
        for inner_index, split in enumerate(inner_splits):
            validation = training.iloc[split.test].reset_index(drop=True)
            cache_path = output / f"base_{cache_key}_{inner_index}.npz"
            if cache_path.exists():
                with np.load(cache_path) as stored:
                    np.testing.assert_array_equal(
                        stored["ids"], validation.unique_trial_id.astype(str)
                    )
                    prediction = stored["prediction"]
            else:
                fitted = ForestModel(
                    specification, seed + inner_index, threads
                ).fit(feature_matrix[split.train], training.label.iloc[split.train])
                prediction = fitted.predict(feature_matrix[split.test])
                np.savez_compressed(
                    cache_path,
                    ids=validation.unique_trial_id.to_numpy(dtype=str),
                    prediction=prediction,
                )
            scores.append(evaluate(validation, prediction, split.regime))
        base_records.append(
            {
                "key": specification["key"],
                "spec": specification,
                "mean": mean_metrics(scores),
                "cells": scores,
            }
        )
        write_json(output / "base_scores.json", base_records)
        write_json(
            output / "progress.json",
            {
                "phase": "base_selection",
                "completed": specification_index + 1,
                "total": len(specifications),
            },
        )
    selected_base = min(base_records, key=rank_record)
    specification = selected_base["spec"]
    feature_matrix = local_views[specification["view"]]

    penalty_scores = {
        (reader_penalty, question_penalty): []
        for reader_penalty in PENALTIES
        for question_penalty in PENALTIES
    }
    cache_key = object_id(specification)
    for inner_index, split in enumerate(inner_splits):
        fit_metadata = training.iloc[split.train].reset_index(drop=True)
        validation = training.iloc[split.test].reset_index(drop=True)
        with np.load(output / f"base_{cache_key}_{inner_index}.npz") as stored:
            base_prediction = stored["prediction"]
        oof_path = output / f"offset_oof_{cache_key}_{inner_index}.npz"
        if oof_path.exists():
            with np.load(oof_path) as stored:
                np.testing.assert_array_equal(
                    stored["ids"], fit_metadata.unique_trial_id.astype(str)
                )
                text_oof = stored["text_oof"]
                reader_oof = stored["reader_oof"]
        else:
            text_oof, text_fold = grouped_oof_predictions(
                fit_metadata,
                feature_matrix[split.train],
                specification,
                seed + 1000 + inner_index,
                "material_id",
                threads,
            )
            reader_oof, reader_fold_assignment = grouped_oof_predictions(
                fit_metadata,
                feature_matrix[split.train],
                specification,
                seed + 1000 + inner_index,
                "participant_id",
                threads,
            )
            np.savez_compressed(
                oof_path,
                ids=fit_metadata.unique_trial_id.to_numpy(dtype=str),
                text_oof=text_oof,
                reader_oof=reader_oof,
                text_fold=text_fold,
                reader_fold=reader_fold_assignment,
            )
        for reader_penalty, question_penalty in penalty_scores:
            correction = PairedOffsets(reader_penalty, question_penalty).fit(
                fit_metadata, text_oof, reader_oof
            )
            prediction = correction.predict(validation, base_prediction)
            penalty_scores[reader_penalty, question_penalty].append(
                evaluate(validation, prediction, split.regime)
            )
        write_json(
            output / "progress.json",
            {
                "phase": "penalty_selection",
                "completed": inner_index + 1,
                "total": len(inner_splits),
            },
        )
    penalty_records = [
        {
            "key": f"offset/{reader_penalty}/{question_penalty}",
            "reader_penalty": reader_penalty,
            "question_penalty": question_penalty,
            "mean": mean_metrics(scores),
            "cells": scores,
        }
        for (reader_penalty, question_penalty), scores in penalty_scores.items()
    ]
    selected_penalty = min(penalty_records, key=rank_record)
    selection = {
        "base": selected_base,
        "offset": selected_penalty,
        "base_selected_before_offset": True,
    }
    write_json(output / "penalty_scores.json", penalty_records)
    write_json(output / "selection.json", selection)

    target = assigned.iloc[outer.test].reset_index(drop=True)
    fitted = ForestModel(specification, seed, threads).fit(
        feature_matrix, training.label
    )
    base_prediction = fitted.predict(views[specification["view"]][outer.test])
    text_oof, text_fold = grouped_oof_predictions(
        training,
        feature_matrix,
        specification,
        seed + 1000,
        "material_id",
        threads,
    )
    reader_oof, reader_fold_assignment = grouped_oof_predictions(
        training,
        feature_matrix,
        specification,
        seed + 1000,
        "participant_id",
        threads,
    )
    correction = PairedOffsets(
        selected_penalty["reader_penalty"], selected_penalty["question_penalty"]
    ).fit(training, text_oof, reader_oof)
    augmented_prediction = correction.predict(target, base_prediction)
    prediction_path = output / "predictions_unscored.npz"
    np.savez_compressed(
        prediction_path,
        ids=target.unique_trial_id.to_numpy(dtype=str),
        regime=outer.regime,
        base=base_prediction,
        augmented=augmented_prediction,
        text_fold=text_fold,
        reader_fold=reader_fold_assignment,
    )
    scores = {
        "base": evaluate(target, base_prediction, outer.regime),
        "augmented": evaluate(target, augmented_prediction, outer.regime),
    }
    write_json(output / "metrics.json", scores)
    write_json(
        completion,
        {
            "rows": len(target),
            "prediction_sha256": sha256(prediction_path),
            "offset_gradient_max": correction.gradient_max,
        },
    )
    return output.name


def load_cell(
    assigned: pd.DataFrame, cell_folder: Path, reader_fold: int, material_fold: int
) -> tuple[dict, pd.DataFrame]:
    outer = _find_outer(assigned, reader_fold, material_fold)
    target = assigned.iloc[outer.test].reset_index(drop=True)
    completion = read_json(cell_folder / "complete.json")
    prediction_path = cell_folder / "predictions_unscored.npz"
    if sha256(prediction_path) != completion["prediction_sha256"]:
        raise ValueError(f"Prediction checksum mismatch: {cell_folder}")
    with np.load(prediction_path) as stored:
        np.testing.assert_array_equal(stored["ids"], target.unique_trial_id.astype(str))
        np.testing.assert_array_equal(stored["regime"], outer.regime)
        scores = {
            "base": evaluate(target, stored["base"], outer.regime),
            "augmented": evaluate(target, stored["augmented"], outer.regime),
        }
        frame = target[
            ["unique_trial_id", "participant_id", "item_key", "material_id", "label"]
        ].copy()
        frame["cell"] = cell_folder.name
        frame["regime"] = outer.regime
        frame["base"] = stored["base"]
        frame["augmented"] = stored["augmented"]
    expected = read_json(cell_folder / "metrics.json")
    for model in scores:
        _metric_match(scores[model], expected[model], f"{cell_folder.name}/{model}")
    selection = read_json(cell_folder / "selection.json")
    if not selection["base_selected_before_offset"]:
        raise ValueError(f"Base-first selection assertion failed: {cell_folder.name}")
    return {"scores": scores, "selection": selection}, frame


def summarize_cells(
    assigned: pd.DataFrame, root: Path, dataset: str
) -> tuple[dict, list[pd.DataFrame]]:
    records = []
    frames = []
    for split in cartesian_splits(assigned):
        name = f"{dataset}_r{split.reader_fold}_m{split.material_fold}"
        record, frame = load_cell(
            assigned, root / name, split.reader_fold, split.material_fold
        )
        records.append(record)
        frames.append(frame)
    means = {
        model: mean_metrics([record["scores"][model] for record in records])
        for model in ("base", "augmented")
    }
    gain = {
        metric: float(
            np.mean(
                [
                    record["scores"]["augmented"][metric]
                    - record["scores"]["base"][metric]
                    for record in records
                ]
            )
        )
        for metric in means["base"]
    }
    family_counts: dict[str, int] = {}
    view_counts: dict[str, int] = {}
    for record in records:
        specification = record["selection"]["base"]["spec"]
        family_counts[specification["family"]] = family_counts.get(
            specification["family"], 0
        ) + 1
        view_counts[specification["view"]] = view_counts.get(
            specification["view"], 0
        ) + 1
    return {
        "means": means,
        "augmented_minus_base": gain,
        "selected_families": family_counts,
        "selected_views": view_counts,
        "cells": len(records),
    }, frames
