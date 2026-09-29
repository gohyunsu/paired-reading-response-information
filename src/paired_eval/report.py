from __future__ import annotations

from pathlib import Path

from .utils import read_json, sha256, write_json


EXPECTED_TITLE = (
    "Paired Evaluation of Reader- and Question-Level Response Information "
    "for Eye-Tracking-Based Reading-Comprehension Prediction"
)
EXPECTED_AUTHORS = (
    "Hyunsu Go",
    "Sumin Lee",
    "Kyeonghun Kim",
    "Subeen Lee",
    "Eunseob Choi",
    "Ken Ying-Kai Liao",
    "Nam-Joon Kim",
)


def _validate_reported_values(result: dict) -> None:
    primary = {
        (row["dataset"], row["regime"]): row for row in result["table_i"]
    }
    expected_primary = {
        ("OneStop", "UT"): (0.561, 5.9, [0.030, 0.090]),
        ("OneStop", "UR"): (0.621, 7.8, [0.055, 0.102]),
        ("SB-SAT", "UT"): (0.564, 1.0, None),
        ("SB-SAT", "UR"): (0.542, 15.3, [0.072, 0.226]),
    }
    for key, (base, gain, interval) in expected_primary.items():
        row = primary.get(key, {})
        if round(row.get("base_auroc", -1), 3) != base:
            raise ValueError(f"Primary base AUROC changed for {key}")
        if row.get("paper_gain_pp") != gain:
            raise ValueError(f"Primary gain changed for {key}")
        if row.get("interval_95_gain_auroc") != interval:
            raise ValueError(f"Primary interval changed for {key}")

    controlled = {
        (row["dataset"], row["arm"]): row for row in result["table_ii"]
    }
    expected_controlled = {
        ("OneStop", "joint"): (7.7, 6.0),
        ("OneStop", "question"): (1.8, 6.0),
        ("OneStop", "question_options"): (0.3, 6.0),
        ("SB-SAT", "scanpath"): (15.6, 0.5),
        ("SB-SAT", "lexical"): (0.5, 3.2),
    }
    for key, gains in expected_controlled.items():
        row = controlled.get(key, {})
        if (row.get("paper_gain_ur_pp"), row.get("paper_gain_ut_pp")) != gains:
            raise ValueError(f"Controlled-representation gains changed for {key}")


def summarize(
    primary: Path, representations: Path, prequential: Path, output: Path
) -> dict:
    report = {
        "primary": read_json(primary / "evaluation.json"),
        "representations": read_json(representations / "evaluation.json"),
        "prequential": read_json(prequential / "evaluation.json"),
    }
    write_json(output, report)
    return report


def verify_release(root: Path | None = None) -> dict:
    if root is None:
        root = Path(__file__).resolve().parents[2]
    result_path = root / "results" / "paper_results.json"
    result = read_json(result_path)

    publication = result.get("publication", {})
    if publication.get("title") != EXPECTED_TITLE:
        raise ValueError("Publication title does not match the release contract")
    if tuple(publication.get("authors", ())) != EXPECTED_AUTHORS:
        raise ValueError("Publication author list does not match the release contract")
    if publication.get("year") != 2026:
        raise ValueError("Publication year does not match the release contract")
    if publication.get("venue") != "IEEE International Conference on Consumer Electronics--Asia":
        raise ValueError("Publication venue does not match the release contract")
    _validate_reported_values(result)

    robustness = result.get("robustness", {})
    if robustness.get("sbsat_reader_partition_by_model_family") != {
        "comparisons": 9,
        "positive_paired_gain_in_all": True,
    }:
        raise ValueError("SB-SAT robustness contract changed")
    if robustness.get("onestop_official_fold_regime_comparisons") != {
        "comparisons": 10,
        "positive_paired_gain_in_all": True,
    }:
        raise ValueError("OneStop official-fold robustness contract changed")
    alternative = robustness.get("alternative_oof_and_selection_objectives", {})
    if alternative.get("maximum_absolute_difference_pp") != 0.5 or not alternative.get(
        "all_intervals_include_zero"
    ):
        raise ValueError("Alternative-construction robustness contract changed")

    excluded = {".git", ".venv", "venv", "outputs", "data"}
    pdf_files = [
        path
        for path in root.rglob("*.pdf")
        if not excluded.intersection(path.relative_to(root).parts)
    ]
    if pdf_files:
        relative = [str(path.relative_to(root)) for path in pdf_files]
        raise ValueError(f"PDF files are not part of this repository: {relative}")

    asset_manifest = read_json(root / "assets" / "manifest.json")
    expected_figures = {
        "figure1_framework.png",
        "figure2_inputs.png",
        "figure3_prequential.png",
    }
    if set(asset_manifest.get("figures", {})) != expected_figures:
        raise ValueError("README figure set changed")
    for filename, record in asset_manifest["figures"].items():
        if sha256(root / "assets" / filename) != record["sha256"]:
            raise ValueError(f"README figure checksum mismatch: {filename}")
    summary = asset_manifest["summary_figure"]
    summary_path = root / "assets" / summary["filename"]
    if summary["source_sha256"] != sha256(result_path):
        raise ValueError("README summary was rendered from different results")
    if summary["sha256"] != sha256(summary_path):
        raise ValueError("README summary checksum mismatch")

    readme = (root / "README.md").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if ".pdf" in readme.lower() or ".pdf" in citation.lower():
        raise ValueError("Public project metadata contains a PDF link")
    for author in EXPECTED_AUTHORS:
        if author not in readme and author.replace(" ", ", ", 1) not in readme:
            raise ValueError(f"README is missing publication author {author}")
    if "Jiwon Yang" in readme or "given-names: Jiwon" in citation:
        raise ValueError("Public metadata contains an unexpected author")

    expected_counts = (
        result["datasets"]["OneStop"]["rows"]
        + result["datasets"]["SB-SAT"]["rows"],
        result["datasets"]["OneStop"]["readers"]
        + result["datasets"]["SB-SAT"]["readers"],
        result["datasets"]["OneStop"]["texts"]
        + result["datasets"]["SB-SAT"]["texts"],
    )
    expected_tokens = (
        f"**{expected_counts[0]:,} responses**",
        f"**{expected_counts[1]} readers**",
        f"**{expected_counts[2]} texts**",
    )
    for token in expected_tokens:
        if token not in readme:
            raise ValueError(f"README cohort summary is missing {token}")

    return {
        "status": "pass",
        "result_manifest_sha256": sha256(result_path),
        "readme_figures": len(asset_manifest["figures"]) + 1,
        "pdf_files": 0,
    }
