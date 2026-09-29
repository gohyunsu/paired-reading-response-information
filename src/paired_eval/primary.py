from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, load_dataset
from .design import OUTER_FOLDS
from .models import primary_candidates
from .nested import run_nested_cell, summarize_cells
from .splits import cartesian_splits
from .uncertainty import crossed_gain_bootstrap
from .utils import write_json


PRIMARY_VIEWS = {
    "OneStop_RC": ("text", "gaze", "joint"),
    "SBSAT_RC": ("compact", "scanpath"),
}


def _load_inputs(bundle: Path, design: Path, dataset: str):
    metadata, views = load_dataset(bundle, dataset)
    assigned = pd.read_csv(design / dataset / "outer_assignment.csv")
    np.testing.assert_array_equal(
        metadata.unique_trial_id.astype(str), assigned.unique_trial_id.astype(str)
    )
    np.testing.assert_array_equal(metadata.label, assigned.label)
    retained = {name: views[name] for name in PRIMARY_VIEWS[dataset]}
    return assigned, retained


def _run_job(
    bundle: Path,
    design: Path,
    output: Path,
    dataset: str,
    reader_fold: int,
    material_fold: int,
    threads: int,
) -> str:
    assigned, views = _load_inputs(bundle, design, dataset)
    specifications = primary_candidates(PRIMARY_VIEWS[dataset])
    expected = 39 if dataset == "OneStop_RC" else 26
    if len(specifications) != expected:
        raise AssertionError(f"Expected {expected} primary candidates")
    cell = output / dataset / f"{dataset}_r{reader_fold}_m{material_fold}"
    return run_nested_cell(
        assigned,
        views,
        dataset,
        reader_fold,
        material_fold,
        specifications,
        cell,
        threads=threads,
    )


def run_primary(
    bundle: Path,
    design: Path,
    output: Path,
    *,
    workers: int = 1,
    threads: int = 1,
    bootstrap_repetitions: int = 2000,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in DATASETS:
        (output / dataset).mkdir(exist_ok=True)
        assigned = pd.read_csv(design / dataset / "outer_assignment.csv")
        for split in cartesian_splits(assigned):
            jobs.append((dataset, split.reader_fold, split.material_fold))
    if workers == 1:
        for dataset, reader_fold, material_fold in jobs:
            _run_job(
                bundle,
                design,
                output,
                dataset,
                reader_fold,
                material_fold,
                threads,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_job,
                    bundle,
                    design,
                    output,
                    dataset,
                    reader_fold,
                    material_fold,
                    threads,
                )
                for dataset, reader_fold, material_fold in jobs
            ]
            for future in as_completed(futures):
                future.result()
    return summarize_primary(
        design, output, bootstrap_repetitions=bootstrap_repetitions
    )


def summarize_primary(
    design: Path, output: Path, *, bootstrap_repetitions: int = 2000
) -> dict:
    report = {
        "protocol": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds_per_axis": 3,
            "candidate_counts": {"OneStop_RC": 39, "SBSAT_RC": 26},
            "selection": "base predictor first; response-offset penalties second",
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        assigned = pd.read_csv(design / dataset / "outer_assignment.csv")
        summary, frames = summarize_cells(assigned, output / dataset, dataset)
        intervals = {}
        for regime, offset in (("UT", 1), ("UR", 2)):
            base_seed = 20260929 if dataset == "OneStop_RC" else 20260939
            intervals[regime] = crossed_gain_bootstrap(
                {"primary": frames},
                regime=regime,
                repetitions=bootstrap_repetitions,
                seed=base_seed + offset,
                minimum_valid=max(1000, bootstrap_repetitions // 2),
            )
        report["datasets"][dataset] = {
            **summary,
            "gain_bootstrap": intervals,
        }
    write_json(output / "evaluation.json", report)
    return report
