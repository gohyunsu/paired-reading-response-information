from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DATASETS,
    load_dataset,
    onestop_representation_views,
    sbsat_lexical_descriptors,
)
from .models import controlled_candidates
from .nested import run_nested_cell, summarize_cells
from .splits import cartesian_splits
from .uncertainty import crossed_gain_bootstrap
from .utils import write_json


ARMS = {
    "OneStop_RC": ("joint", "question", "question_options"),
    "SBSAT_RC": ("scanpath", "lexical"),
}


def _load_arm(
    bundle: Path,
    design: Path,
    dataset: str,
    arm: str,
    onestop_qa: Path,
    sbsat_stimuli: Path,
):
    metadata, base = load_dataset(bundle, dataset)
    assigned = pd.read_csv(design / dataset / "outer_assignment.csv")
    np.testing.assert_array_equal(
        metadata.unique_trial_id.astype(str), assigned.unique_trial_id.astype(str)
    )
    np.testing.assert_array_equal(metadata.label, assigned.label)
    if dataset == "OneStop_RC":
        views = onestop_representation_views(assigned, base["joint"], onestop_qa)
        matrix = views[arm]
    elif arm == "scanpath":
        matrix = base["scanpath"]
    else:
        lexical = sbsat_lexical_descriptors(assigned, sbsat_stimuli)
        matrix = np.c_[base["scanpath"], lexical].astype(np.float32)
    return assigned, {arm: matrix}


def _run_job(
    bundle: Path,
    design: Path,
    onestop_qa: Path,
    sbsat_stimuli: Path,
    output: Path,
    dataset: str,
    arm: str,
    reader_fold: int,
    material_fold: int,
    threads: int,
) -> str:
    assigned, views = _load_arm(
        bundle, design, dataset, arm, onestop_qa, sbsat_stimuli
    )
    cell = output / dataset / arm / f"{dataset}_r{reader_fold}_m{material_fold}"
    return run_nested_cell(
        assigned,
        views,
        dataset,
        reader_fold,
        material_fold,
        controlled_candidates(arm),
        cell,
        threads=threads,
    )


def run_representations(
    bundle: Path,
    design: Path,
    onestop_qa: Path,
    sbsat_stimuli: Path,
    output: Path,
    *,
    workers: int = 1,
    threads: int = 1,
    bootstrap_repetitions: int = 2000,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, arms in ARMS.items():
        assigned = pd.read_csv(design / dataset / "outer_assignment.csv")
        for arm in arms:
            (output / dataset / arm).mkdir(parents=True, exist_ok=True)
            for split in cartesian_splits(assigned):
                jobs.append((dataset, arm, split.reader_fold, split.material_fold))
    if workers == 1:
        for dataset, arm, reader_fold, material_fold in jobs:
            _run_job(
                bundle,
                design,
                onestop_qa,
                sbsat_stimuli,
                output,
                dataset,
                arm,
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
                    onestop_qa,
                    sbsat_stimuli,
                    output,
                    dataset,
                    arm,
                    reader_fold,
                    material_fold,
                    threads,
                )
                for dataset, arm, reader_fold, material_fold in jobs
            ]
            for future in as_completed(futures):
                future.result()
    return summarize_representations(
        design, output, bootstrap_repetitions=bootstrap_repetitions
    )


def summarize_representations(
    design: Path, output: Path, *, bootstrap_repetitions: int = 2000
) -> dict:
    report = {
        "protocol": {
            "candidate_families": ["RF", "ExtraTrees"],
            "trees": 32,
            "max_depth": 12,
            "minimum_leaf_size": 5,
            "feature_fraction": 0.5,
            "selection": "arm-specific base selection followed by penalty selection",
        },
        "datasets": {},
    }
    for dataset, arms in ARMS.items():
        assigned = pd.read_csv(design / dataset / "outer_assignment.csv")
        summaries = {}
        frames = {}
        for arm in arms:
            summaries[arm], frames[arm] = summarize_cells(
                assigned, output / dataset / arm, dataset
            )
        seed = 20261511 if dataset == "OneStop_RC" else 20261512
        bootstrap = crossed_gain_bootstrap(
            frames,
            regime="UR",
            repetitions=bootstrap_repetitions,
            seed=seed,
            minimum_valid=max(1000, bootstrap_repetitions // 2),
        )
        report["datasets"][dataset] = {
            "arms": summaries,
            "ur_gain_bootstrap": bootstrap,
        }
    write_json(output / "evaluation.json", report)
    return report
