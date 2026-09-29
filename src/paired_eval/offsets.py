from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

from .models import ForestModel


PENALTIES = (0.1, 1.0, 10.0, 100.0)


def solve_offset_logistic(
    design,
    offsets,
    labels,
    penalty,
    *,
    gradient_tolerance: float = 1e-8,
    max_iterations: int = 100,
):
    """Minimize summed Bernoulli loss plus a diagonal L2 penalty."""
    design = csr_matrix(design, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    labels = np.asarray(labels, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    observations, dimensions = design.shape
    if offsets.shape != (observations,) or labels.shape != (observations,):
        raise ValueError("Design and response vectors have incompatible shapes")
    if penalty.shape != (dimensions,) or penalty[0] != 0 or not (penalty[1:] > 0).all():
        raise ValueError("Only the intercept may be unpenalized")
    if not np.array_equal(design[:, 0].toarray().ravel(), np.ones(observations)):
        raise ValueError("The first design column must be an intercept")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("Both binary outcomes are required")
    parameter = np.zeros(dimensions, dtype=float)

    def objective(value):
        linear = offsets + design @ value
        loss = np.logaddexp(0.0, (1 - 2 * labels) * linear).sum()
        loss += 0.5 * np.dot(penalty, value**2)
        gradient = np.asarray(design.T @ (expit(linear) - labels)).ravel()
        gradient += penalty * value
        return float(loss), gradient

    for iteration in range(max_iterations + 1):
        loss, gradient = objective(parameter)
        norm = float(np.max(np.abs(gradient)))
        if norm <= gradient_tolerance:
            return parameter, norm, iteration
        if iteration == max_iterations:
            break
        probability = expit(offsets + design @ parameter)
        hessian = design.T @ design.multiply(
            (probability * (1 - probability))[:, None]
        ) + diags(penalty)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            direction = spsolve(hessian.tocsc(), gradient)
        decrement = float(gradient @ direction)
        if not np.isfinite(direction).all() or decrement <= 0:
            raise RuntimeError("Invalid Newton direction")
        scale = 1.0
        for _ in range(60):
            candidate = parameter - scale * direction
            candidate_loss, candidate_gradient = objective(candidate)
            resolution = 16 * np.finfo(float).eps * max(1.0, abs(loss))
            armijo = candidate_loss <= loss - 1e-4 * scale * decrement
            near_precision = (
                abs(candidate_loss - loss) <= resolution
                and np.max(np.abs(candidate_gradient)) < norm
            )
            if np.isfinite(candidate_loss) and (armijo or near_precision):
                parameter = candidate
                break
            scale *= 0.5
        else:
            raise RuntimeError("Newton line search failed")
    raise RuntimeError("Offset solver did not reach gradient stationarity")


class PairedOffsets:
    """Shared-intercept reader and question offsets from two OOF score views."""

    def __init__(self, reader_penalty: float, question_penalty: float):
        self.reader_penalty = float(reader_penalty)
        self.question_penalty = float(question_penalty)

    def fit(
        self,
        metadata: pd.DataFrame,
        text_held_out_scores: np.ndarray,
        reader_held_out_scores: np.ndarray,
    ) -> "PairedOffsets":
        self.readers = {
            value: index
            for index, value in enumerate(sorted(metadata.participant_id.unique()))
        }
        self.questions = {
            value: index for index, value in enumerate(sorted(metadata.item_key.unique()))
        }
        observations = len(metadata)
        reader_count = len(self.readers)
        question_count = len(self.questions)
        reader_column = metadata.participant_id.map(self.readers).to_numpy(int) + 1
        question_column = (
            metadata.item_key.map(self.questions).to_numpy(int) + 1 + reader_count
        )
        rows = list(range(2 * observations))
        columns = [0] * (2 * observations)
        rows.extend(range(2 * observations))
        columns.extend(np.r_[reader_column, question_column])
        design = csr_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(2 * observations, 1 + reader_count + question_count),
        )
        score_offset = logit(
            np.clip(
                np.r_[text_held_out_scores, reader_held_out_scores], 1e-5, 1 - 1e-5
            )
        )
        labels = np.tile(metadata.label.to_numpy(float), 2)
        penalty = np.r_[
            0.0,
            np.full(reader_count, self.reader_penalty),
            np.full(question_count, self.question_penalty),
        ]
        self.parameter, self.gradient_max, self.iterations = solve_offset_logistic(
            design, score_offset, labels, penalty
        )
        return self

    def components(self, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        reader_count = len(self.readers)
        reader_map = {
            value: self.parameter[1 + index] for value, index in self.readers.items()
        }
        question_map = {
            value: self.parameter[1 + reader_count + index]
            for value, index in self.questions.items()
        }
        reader = metadata.participant_id.map(reader_map).fillna(0.0).to_numpy(float)
        question = metadata.item_key.map(question_map).fillna(0.0).to_numpy(float)
        return reader, question

    def predict(self, metadata: pd.DataFrame, base_scores: np.ndarray) -> np.ndarray:
        reader, question = self.components(metadata)
        base_logit = logit(np.clip(base_scores, 1e-5, 1 - 1e-5))
        return expit(base_logit + self.parameter[0] + reader + question)


def grouped_oof_predictions(
    metadata: pd.DataFrame,
    features: np.ndarray,
    specification: dict,
    seed: int,
    group: str,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    folds = min(3, metadata[group].nunique())
    if folds < 2:
        raise ValueError(f"Insufficient {group} groups for OOF prediction")
    prediction = np.full(len(metadata), np.nan)
    assignment = np.full(len(metadata), -1, dtype=int)
    splitter = GroupKFold(n_splits=folds)
    for index, (train, valid) in enumerate(
        splitter.split(features, groups=metadata[group])
    ):
        if not set(metadata.iloc[train][group]).isdisjoint(metadata.iloc[valid][group]):
            raise AssertionError(f"{group} crosses an OOF fold")
        model = ForestModel(specification, seed + index, threads).fit(
            features[train], metadata.label.iloc[train]
        )
        prediction[valid] = model.predict(features[valid])
        assignment[valid] = index
    if not np.isfinite(prediction).all() or (assignment < 0).any():
        raise AssertionError("Incomplete OOF predictions")
    return prediction, assignment


def fit_scalar_offset(base_scores, labels, penalty: float) -> float:
    base_scores = np.asarray(base_scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if not len(labels):
        return 0.0
    offset = logit(np.clip(base_scores, 1e-5, 1 - 1e-5))
    value = 0.0
    for _ in range(50):
        probability = expit(offset + value)
        gradient = float(np.sum(probability - labels) + penalty * value)
        hessian = float(np.sum(probability * (1 - probability)) + penalty)
        step = gradient / hessian
        value -= step
        if abs(step) < 1e-12:
            break
    return value
