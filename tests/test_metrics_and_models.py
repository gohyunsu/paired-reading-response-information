import numpy as np
import pandas as pd

from paired_eval.data import sequence_descriptor
from paired_eval.metrics import evaluate, within_question_auc
from paired_eval.models import controlled_candidates, primary_candidates


def test_candidate_counts_match_manuscript():
    assert len(primary_candidates(("text", "gaze", "joint"))) == 39
    assert len(primary_candidates(("compact", "scanpath"))) == 26
    assert len(controlled_candidates("joint")) == 2


def test_sequence_descriptor_dimension_and_finiteness():
    rng = np.random.default_rng(4)
    sequence = rng.normal(size=(25, 6))
    descriptor = sequence_descriptor(sequence)
    assert descriptor.shape == (142,)
    assert np.isfinite(descriptor).all()


def test_metrics_are_regime_specific():
    metadata = pd.DataFrame(
        {
            "item_key": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    regime = np.array(["UT", "UT", "UR", "UR", "UB", "UB"])
    result = evaluate(metadata, scores, regime)
    assert result["UT"] == result["UR"] == result["UB"] == 1.0
    assert result["macro"] == 1.0
    assert within_question_auc(metadata, scores) == 1.0
