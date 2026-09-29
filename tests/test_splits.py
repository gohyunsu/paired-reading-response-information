import numpy as np
import pandas as pd

from paired_eval.splits import assign_groups, cartesian_splits, validate_coverage


def synthetic_metadata():
    rows = []
    for reader in range(8):
        for material in range(4):
            for question in range(2):
                rows.append(
                    {
                        "unique_trial_id": f"r{reader}_m{material}_q{question}",
                        "participant_id": f"r{reader}",
                        "unique_paragraph_id": f"story{material}",
                        "item_key": f"story{material}_q{question}",
                        "label": (reader + material + question) % 2,
                        "scan_group": f"r{reader}::story{material}",
                    }
                )
    return pd.DataFrame(rows)


def test_cartesian_split_support_and_coverage():
    metadata = synthetic_metadata()
    assigned = assign_groups(metadata, "SBSAT_RC", folds=4, seed=31)
    splits = cartesian_splits(assigned)
    assert len(splits) == 16
    validate_coverage(assigned, splits, folds=4)
    for split in splits:
        assert set(split.regime) == {"UT", "UR", "UB"}
        assert set(assigned.iloc[split.train].scan_group).isdisjoint(
            assigned.iloc[split.test].scan_group
        )


def test_assignment_is_label_independent():
    metadata = synthetic_metadata()
    first = assign_groups(metadata, "SBSAT_RC", folds=4, seed=31)
    shuffled_labels = metadata.copy()
    shuffled_labels["label"] = np.random.default_rng(8).permutation(
        shuffled_labels.label
    )
    second = assign_groups(shuffled_labels, "SBSAT_RC", folds=4, seed=31)
    np.testing.assert_array_equal(first.reader_fold, second.reader_fold)
    np.testing.assert_array_equal(first.material_fold, second.material_fold)
