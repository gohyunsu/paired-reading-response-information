"""Build the README summary graphic and asset checksum manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper_results.json"
OUTPUT = ROOT / "assets"
FIGURES = {
    "figure1_framework.png": "Paired evaluation framework",
    "figure2_inputs.png": "SB-SAT trial construction and inputs",
    "figure3_prequential.png": "Prequential reader adaptation",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_svg(results: dict) -> str:
    primary = {
        (row["dataset"], row["regime"]): row for row in results["table_i"]
    }
    controlled = {
        (row["dataset"], row["arm"]): row for row in results["table_ii"]
    }
    question_primary = (
        primary[("OneStop", "UR")]["paper_gain_pp"],
        primary[("SB-SAT", "UR")]["paper_gain_pp"],
    )
    question_controlled = (
        controlled[("OneStop", "question_options")]["paper_gain_ur_pp"],
        controlled[("SB-SAT", "lexical")]["paper_gain_ur_pp"],
    )
    penalties = results["prequential"]["penalties"].values()
    if not all(row["pooled_gain"] > 0 and row["within_reader_gain"] < 0 for row in penalties):
        raise ValueError("Prequential result direction changed")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360" viewBox="0 0 1200 360" role="img" aria-labelledby="title desc">
  <title id="title">Results at a glance</title>
  <desc id="desc">Three findings: question-level response information produces large gains, question-varying inputs reduce those gains, and prequential reader adaptation improves pooled but not within-reader AUROC.</desc>
  <defs>
    <style>
      .heading {{ font: 600 24px Arial, Helvetica, sans-serif; fill: #202631; }}
      .value {{ font: 600 38px Arial, Helvetica, sans-serif; fill: #235a85; }}
      .label {{ font: 18px Arial, Helvetica, sans-serif; fill: #3d4652; }}
      .small {{ font: 16px Arial, Helvetica, sans-serif; fill: #5c6673; }}
      .card {{ fill: #ffffff; stroke: #c9d0d8; stroke-width: 1.5; }}
      .rule {{ stroke: #d9dee4; stroke-width: 1.5; }}
      .accent {{ stroke: #235a85; stroke-width: 5; stroke-linecap: round; }}
      .muted {{ stroke: #8c98a5; stroke-width: 5; stroke-linecap: round; }}
    </style>
  </defs>
  <rect width="1200" height="360" rx="12" fill="#f5f7f9"/>
  <text x="60" y="54" class="heading">Results at a glance</text>

  <rect x="48" y="82" width="344" height="230" rx="10" class="card"/>
  <line x1="76" y1="111" x2="124" y2="111" class="accent"/>
  <text x="76" y="153" class="value">+{question_primary[0]:.1f} / +{question_primary[1]:.1f} pp</text>
  <text x="76" y="190" class="label">Question-level gain</text>
  <line x1="76" y1="212" x2="364" y2="212" class="rule"/>
  <text x="76" y="245" class="small">OneStop / SB-SAT</text>
  <text x="76" y="273" class="small">Primary unseen-reader evaluation</text>

  <rect x="428" y="82" width="344" height="230" rx="10" class="card"/>
  <line x1="456" y1="111" x2="504" y2="111" class="accent"/>
  <text x="456" y="153" class="value">{question_controlled[0]:.1f} / {question_controlled[1]:.1f} pp</text>
  <text x="456" y="190" class="label">Gain after richer inputs</text>
  <line x1="456" y1="212" x2="744" y2="212" class="rule"/>
  <text x="456" y="245" class="small">OneStop / SB-SAT</text>
  <text x="456" y="273" class="small">Matched-capacity comparison</text>

  <rect x="808" y="82" width="344" height="230" rx="10" class="card"/>
  <line x1="836" y1="111" x2="884" y2="111" class="muted"/>
  <text x="836" y="151" class="heading">Reader adaptation</text>
  <text x="836" y="190" class="label">Pooled AUROC ↑</text>
  <line x1="836" y1="212" x2="1124" y2="212" class="rule"/>
  <text x="836" y="245" class="label">Within-reader AUROC ↓</text>
  <text x="836" y="277" class="small">Across four fixed penalties</text>
</svg>
'''


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"figures": {}}
    for filename, description in FIGURES.items():
        destination = OUTPUT / filename
        if not destination.is_file():
            raise FileNotFoundError(destination)
        manifest["figures"][filename] = {
            "description": description,
            "sha256": digest(destination),
        }

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    summary = OUTPUT / "results_at_a_glance.svg"
    summary.write_text(_summary_svg(results), encoding="utf-8")
    manifest["summary_figure"] = {
        "filename": summary.name,
        "source": str(RESULTS.relative_to(ROOT)),
        "source_sha256": digest(RESULTS),
        "sha256": digest(summary),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
