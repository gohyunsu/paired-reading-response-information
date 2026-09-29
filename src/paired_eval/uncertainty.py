from __future__ import annotations

import numpy as np
import pandas as pd


class WeightedAUC:
    """Efficient AUROC under repeated nonnegative observation weights."""

    def __init__(self, labels, scores, positions):
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)
        positions = np.asarray(positions, dtype=int)
        if len(labels) != len(scores) or len(scores) != len(positions):
            raise ValueError("Labels, scores, and positions must have equal length")
        order = np.argsort(scores, kind="stable")
        self.positions = positions[order]
        self.positive = labels[order].astype(bool)
        self.starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]

    def __call__(self, weights) -> float | None:
        row_weight = np.asarray(weights, dtype=float)[self.positions]
        positive = np.add.reduceat(row_weight * self.positive, self.starts)
        negative = np.add.reduceat(row_weight * ~self.positive, self.starts)
        denominator = positive.sum() * negative.sum()
        if denominator == 0:
            return None
        return float(
            np.dot(positive, np.cumsum(negative) - 0.5 * negative) / denominator
        )


def crossed_gain_bootstrap(
    frames: dict[str, list[pd.DataFrame]],
    *,
    regime: str,
    repetitions: int,
    seed: int,
    minimum_valid: int,
) -> dict:
    """Paired reader-by-text bootstrap of augmented-minus-base AUROC."""
    arms = list(frames)
    if not arms:
        raise ValueError("At least one arm is required")
    reference = arms[0]
    merged = {}
    for arm, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        merged[arm] = frame.loc[frame.regime.eq(regime)].reset_index(drop=True)
    identity = [
        "cell",
        "unique_trial_id",
        "participant_id",
        "material_id",
        "label",
        "regime",
    ]
    for arm in arms[1:]:
        pd.testing.assert_frame_equal(merged[reference][identity], merged[arm][identity])
    base_frame = merged[reference]
    readers = np.sort(base_frame.participant_id.astype(str).unique())
    materials = np.sort(base_frame.material_id.astype(str).unique())
    reader_code = pd.Categorical(
        base_frame.participant_id.astype(str), categories=readers
    ).codes
    material_code = pd.Categorical(
        base_frame.material_id.astype(str), categories=materials
    ).codes
    components: dict[str, list[tuple[WeightedAUC, WeightedAUC]]] = {
        arm: [] for arm in arms
    }
    for _, indices in base_frame.groupby("cell", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        part = base_frame.iloc[positions]
        for arm in arms:
            source = merged[arm].iloc[positions]
            components[arm].append(
                (
                    WeightedAUC(part.label, source.base, positions),
                    WeightedAUC(part.label, source.augmented, positions),
                )
            )

    def aggregate(weights):
        result = {}
        for arm in arms:
            values = []
            for base_component, augmented_component in components[arm]:
                base_auc = base_component(weights)
                augmented_auc = augmented_component(weights)
                if base_auc is None or augmented_auc is None:
                    return None
                values.append(augmented_auc - base_auc)
            result[arm] = float(np.mean(values))
        return result

    point = aggregate(np.ones(len(base_frame)))
    if point is None:
        raise ValueError("Observed gain is undefined")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        reader_multiplicity = rng.multinomial(
            len(readers), np.full(len(readers), 1 / len(readers))
        )
        material_multiplicity = rng.multinomial(
            len(materials), np.full(len(materials), 1 / len(materials))
        )
        weights = (
            reader_multiplicity[reader_code] * material_multiplicity[material_code]
        )
        result = aggregate(weights)
        if result is not None:
            samples.append(result)
    report = {
        "regime": regime,
        "estimand": "equal-cell mean augmented-minus-base AUROC",
        "reference_arm": reference,
        "point": point,
        "attempted": repetitions,
        "valid": len(samples),
        "invalid": repetitions - len(samples),
        "minimum_valid": minimum_valid,
        "seed": seed,
    }
    if len(samples) < minimum_valid:
        report["intervals_suppressed"] = True
        return report
    sample = pd.DataFrame(samples)
    report["intervals_suppressed"] = False
    report["gain_percentile_95"] = {
        arm: sample[arm].quantile([0.025, 0.975]).tolist() for arm in arms
    }
    report["difference_in_gains"] = {}
    for arm in arms[1:]:
        difference = sample[arm] - sample[reference]
        report["difference_in_gains"][f"{arm}_minus_{reference}"] = {
            "point": point[arm] - point[reference],
            "percentile_95": difference.quantile([0.025, 0.975]).tolist(),
        }
    return report
