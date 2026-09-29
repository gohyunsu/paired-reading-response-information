import numpy as np
import pandas as pd
from scipy.special import expit, logit

from paired_eval.offsets import PairedOffsets, fit_scalar_offset, solve_offset_logistic


def test_paired_offsets_preserve_unseen_group_ranking():
    metadata = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "item_key": ["q1", "q2"] * 4,
            "label": [1, 1, 0, 0, 1, 0, 0, 1],
        }
    )
    text_oof = np.array([0.6, 0.7, 0.4, 0.3, 0.7, 0.4, 0.2, 0.8])
    reader_oof = np.array([0.65, 0.65, 0.35, 0.35, 0.55, 0.45, 0.45, 0.55])
    fitted = PairedOffsets(1.0, 1.0).fit(metadata, text_oof, reader_oof)
    target = pd.DataFrame(
        {
            "participant_id": ["unseen", "unseen", "unseen"],
            "item_key": ["new1", "new2", "new3"],
        }
    )
    base = np.array([0.2, 0.5, 0.8])
    prediction = fitted.predict(target, base)
    assert np.array_equal(np.argsort(base), np.argsort(prediction))
    assert fitted.gradient_max <= 1e-8


def test_scalar_offset_uses_observed_history():
    base = np.full(5, 0.5)
    labels = np.ones(5)
    offset = fit_scalar_offset(base, labels, penalty=1.0)
    assert offset > 0
    assert expit(logit(0.5) + offset) > 0.5


def test_solver_stationarity():
    design = np.ones((6, 1))
    parameter, gradient, iterations = solve_offset_logistic(
        design,
        offsets=np.zeros(6),
        labels=np.array([0, 0, 0, 1, 1, 1]),
        penalty=np.array([0.0]),
    )
    assert abs(parameter[0]) < 1e-12
    assert gradient <= 1e-8
    assert iterations == 0
