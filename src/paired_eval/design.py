from __future__ import annotations

from pathlib import Path

from .data import DATASETS, load_dataset
from .metrics import strict_auc
from .splits import assign_groups, cartesian_splits, nested_splits, validate_coverage
from .utils import sha256, write_json


OUTER_FOLDS = {"OneStop_RC": 5, "SBSAT_RC": 4}
OUTER_SEED = 20260929


def build_design(bundle: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for dataset in DATASETS:
        metadata, _ = load_dataset(bundle, dataset)
        folds = OUTER_FOLDS[dataset]
        assigned = assign_groups(metadata, dataset, folds, OUTER_SEED)
        outer_splits = cartesian_splits(assigned)
        validate_coverage(assigned, outer_splits, folds)
        folder = output / dataset
        folder.mkdir()
        assignment_path = folder / "outer_assignment.csv"
        assigned.to_csv(assignment_path, index=False)
        cells = []
        for outer in outer_splits:
            seed = OUTER_SEED + 100 * outer.reader_fold + outer.material_fold
            inner_assigned, inner_splits = nested_splits(
                assigned, dataset, outer, seed, folds=3
            )
            for frame, splits in ((assigned, [outer]), (inner_assigned, inner_splits)):
                for split in splits:
                    for regime in ("UT", "UR", "UB"):
                        labels = frame.iloc[split.test[split.regime == regime]].label
                        strict_auc(labels, [0.5] * len(labels))
            cells.append(
                {
                    "reader_fold": outer.reader_fold,
                    "material_fold": outer.material_fold,
                    "train": len(outer.train),
                    "test": len(outer.test),
                    "regimes": {
                        name: int((outer.regime == name).sum())
                        for name in ("UT", "UR", "UB")
                    },
                    "inner_splits": len(inner_splits),
                }
            )
        report[dataset] = {
            "rows": len(assigned),
            "readers": int(assigned.participant_id.nunique()),
            "texts": int(assigned.material_id.nunique()),
            "outer_splits": len(outer_splits),
            "assignment_sha256": sha256(assignment_path),
            "cells": cells,
        }
    write_json(output / "design.json", report)
    return report
