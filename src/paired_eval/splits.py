from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import material_ids


@dataclass(frozen=True)
class Split:
    reader_fold: int
    material_fold: int
    train: np.ndarray
    test: np.ndarray
    regime: np.ndarray


def assign_groups(metadata: pd.DataFrame, dataset: str, folds: int, seed: int) -> pd.DataFrame:
    """Assign readers and complete texts without consulting response labels."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    assigned = metadata.copy().reset_index(drop=True)
    assigned["material_id"] = material_ids(assigned, dataset)
    for source, destination, offset in (
        ("participant_id", "reader_fold", 0),
        ("material_id", "material_fold", 1009),
    ):
        groups = np.array(sorted(assigned[source].astype(str).unique()))
        if len(groups) < folds:
            raise ValueError(f"Insufficient {source} groups for {folds} folds")
        np.random.default_rng(seed + offset).shuffle(groups)
        if source == "participant_id":
            exposures = assigned[["participant_id", "item_key"]].astype(str).drop_duplicates()
            item_ids = {value: index for index, value in enumerate(sorted(exposures.item_key.unique()))}
            counts = np.zeros((folds, len(item_ids)), dtype=int)
            load = np.zeros(folds, dtype=int)
            mapping: dict[str, int] = {}
            rng = np.random.default_rng(seed + offset + 17)
            reader_items = exposures.groupby("participant_id").item_key.apply(list)
            for reader in groups:
                items = np.array([item_ids[value] for value in reader_items[reader]], dtype=int)
                order = rng.permutation(folds)
                selected = min(
                    order,
                    key=lambda fold: (float(counts[fold, items].mean()), int(load[fold])),
                )
                mapping[str(reader)] = int(selected)
                counts[selected, items] += 1
                load[selected] += 1
        else:
            mapping = {str(value): index % folds for index, value in enumerate(groups)}
        assigned[destination] = assigned[source].astype(str).map(mapping).to_numpy(int)
    return assigned


def cartesian_splits(assigned: pd.DataFrame) -> list[Split]:
    result = []
    for reader_fold in sorted(assigned.reader_fold.unique()):
        for material_fold in sorted(assigned.material_fold.unique()):
            held_reader = assigned.reader_fold.eq(reader_fold).to_numpy()
            held_material = assigned.material_fold.eq(material_fold).to_numpy()
            train = np.flatnonzero(~held_reader & ~held_material)
            test = np.flatnonzero(held_reader | held_material)
            regime = np.where(
                held_reader[test] & held_material[test],
                "UB",
                np.where(held_reader[test], "UR", "UT"),
            )
            split = Split(int(reader_fold), int(material_fold), train, test, regime)
            validate_split(assigned, split)
            result.append(split)
    return result


def validate_split(metadata: pd.DataFrame, split: Split) -> None:
    train = metadata.iloc[split.train]
    target = metadata.iloc[split.test]
    if set(split.train) & set(split.test):
        raise AssertionError("Training and test rows overlap")
    if len(split.train) + len(split.test) != len(metadata):
        raise AssertionError("Split drops observations")
    known_reader = target.participant_id.isin(train.participant_id).to_numpy()
    known_item = target.item_key.isin(train.item_key).to_numpy()
    known_material = target.material_id.isin(train.material_id).to_numpy()
    actual = np.select(
        [known_reader & ~known_item, ~known_reader & known_item, ~known_reader & ~known_item],
        ["UT", "UR", "UB"],
        default="KNOWN",
    )
    np.testing.assert_array_equal(actual, split.regime)
    np.testing.assert_array_equal(known_material, split.regime == "UR")
    if "scan_group" in metadata:
        if not set(train.scan_group).isdisjoint(target.scan_group):
            raise AssertionError("A scanpath crosses the training/test boundary")


def nested_splits(
    assigned: pd.DataFrame, dataset: str, outer: Split, seed: int, folds: int = 3
) -> tuple[pd.DataFrame, list[Split]]:
    training = assigned.iloc[outer.train].reset_index(drop=True)
    inner_assigned = assign_groups(training, dataset, folds, seed)
    return inner_assigned, cartesian_splits(inner_assigned)


def validate_coverage(metadata: pd.DataFrame, splits: list[Split], folds: int) -> None:
    counts = {name: np.zeros(len(metadata), dtype=int) for name in ("UT", "UR", "UB")}
    for split in splits:
        for regime in counts:
            counts[regime][split.test[split.regime == regime]] += 1
    for regime, expected in (("UB", 1), ("UT", folds - 1), ("UR", folds - 1)):
        np.testing.assert_array_equal(counts[regime], np.full(len(metadata), expected))
