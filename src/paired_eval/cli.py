from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import extract_onestop_trial_order
from .design import build_design
from .prequential import run_prequential
from .primary import run_primary, summarize_primary
from .report import summarize, verify_release
from .representations import run_representations, summarize_representations


def _common_compute(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="paired-eval")
    subparsers = root.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser("design", help="Build reader-by-text assignments")
    design.add_argument("--bundle", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)

    order = subparsers.add_parser(
        "extract-trial-order", help="Extract OneStop participant-local trial order"
    )
    order.add_argument("--archive", type=Path, required=True)
    order.add_argument("--metadata", type=Path, required=True)
    order.add_argument("--output", type=Path, required=True)

    primary = subparsers.add_parser("primary", help="Run the primary nested analysis")
    primary.add_argument("--bundle", type=Path, required=True)
    primary.add_argument("--design", type=Path, required=True)
    primary.add_argument("--output", type=Path, required=True)
    primary.add_argument("--bootstrap", type=int, default=2000)
    primary.add_argument("--summarize-only", action="store_true")
    _common_compute(primary)

    representations = subparsers.add_parser(
        "representations", help="Run the controlled representation analysis"
    )
    representations.add_argument("--bundle", type=Path, required=True)
    representations.add_argument("--design", type=Path, required=True)
    representations.add_argument("--onestop-qa", type=Path, required=True)
    representations.add_argument("--sbsat-stimuli", type=Path, required=True)
    representations.add_argument("--output", type=Path, required=True)
    representations.add_argument("--bootstrap", type=int, default=2000)
    representations.add_argument("--summarize-only", action="store_true")
    _common_compute(representations)

    prequential = subparsers.add_parser(
        "prequential", help="Run reader-wise forward evaluation"
    )
    prequential.add_argument("--design", type=Path, required=True)
    prequential.add_argument("--primary", type=Path, required=True)
    prequential.add_argument("--trial-order", type=Path, required=True)
    prequential.add_argument("--output", type=Path, required=True)
    prequential.add_argument("--minimum-history", type=int, default=10)
    prequential.add_argument("--bootstrap", type=int, default=5000)
    prequential.add_argument("--seed", type=int, default=20260926)

    summary = subparsers.add_parser("summarize", help="Combine completed evaluations")
    summary.add_argument("--primary", type=Path, required=True)
    summary.add_argument("--representations", type=Path, required=True)
    summary.add_argument("--prequential", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify-release", help="Verify reported-result and README asset snapshots"
    )
    verify.add_argument("--root", type=Path)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "design":
        result = build_design(arguments.bundle, arguments.output)
    elif arguments.command == "extract-trial-order":
        frame = extract_onestop_trial_order(
            arguments.archive, arguments.metadata, arguments.output
        )
        result = {
            "rows": len(frame),
            "readers": int(frame.participant_id.nunique()),
            "output": str(arguments.output),
        }
    elif arguments.command == "primary":
        if arguments.summarize_only:
            result = summarize_primary(
                arguments.design,
                arguments.output,
                bootstrap_repetitions=arguments.bootstrap,
            )
        else:
            result = run_primary(
                arguments.bundle,
                arguments.design,
                arguments.output,
                workers=arguments.workers,
                threads=arguments.threads,
                bootstrap_repetitions=arguments.bootstrap,
            )
    elif arguments.command == "representations":
        if arguments.summarize_only:
            result = summarize_representations(
                arguments.design,
                arguments.output,
                bootstrap_repetitions=arguments.bootstrap,
            )
        else:
            result = run_representations(
                arguments.bundle,
                arguments.design,
                arguments.onestop_qa,
                arguments.sbsat_stimuli,
                arguments.output,
                workers=arguments.workers,
                threads=arguments.threads,
                bootstrap_repetitions=arguments.bootstrap,
            )
    elif arguments.command == "prequential":
        result = run_prequential(
            arguments.design,
            arguments.primary,
            arguments.trial_order,
            arguments.output,
            minimum_history=arguments.minimum_history,
            bootstrap_repetitions=arguments.bootstrap,
            seed=arguments.seed,
        )
    elif arguments.command == "summarize":
        result = summarize(
            arguments.primary,
            arguments.representations,
            arguments.prequential,
            arguments.output,
        )
    elif arguments.command == "verify-release":
        result = verify_release(arguments.root)
    else:
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
