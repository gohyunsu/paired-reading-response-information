# Reproduction guide

The command-line pipeline separates experimental design, model fitting, analysis, and release verification. Completed outer cells carry input-context and output checksums, allowing interrupted runs to resume without silently reusing incompatible artifacts.

## Requirements

- Python 3.10–3.13
- OneStop ordinary-reading (Gathering) data
- SB-SAT reading-fixation data
- sufficient storage for prepared feature matrices and cell-level predictions

Install the package and test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q
```

## Input preparation

Follow [`data.md`](data.md) to build the feature bundle, OneStop question file, SB-SAT stimulus table, and participant-local OneStop trial order. The loader validates identifiers, dimensions, labels, and group alignment before fitting begins.

## Pipeline

### 1. Freeze the split design

```bash
paired-eval design \
  --bundle /path/to/bundle \
  --output outputs/design
```

This command assigns readers and complete texts to outer and inner folds without using labels. Use the same design directory for every subsequent analysis.

### 2. Run the primary analysis

```bash
paired-eval primary \
  --bundle /path/to/bundle \
  --design outputs/design \
  --output outputs/primary \
  --workers 4 \
  --threads 1
```

The output contains per-cell candidate selection, OOF scores, selected penalties, paired predictions, and a primary summary. Worker and thread counts affect resource use, not the split assignments or candidate definitions.

### 3. Run the controlled representation analysis

```bash
paired-eval representations \
  --bundle /path/to/bundle \
  --design outputs/design \
  --onestop-qa /path/to/onestop_qa.json \
  --sbsat-stimuli /path/to/combined_stimulus.csv \
  --output outputs/representations \
  --workers 4 \
  --threads 1
```

All representation arms reuse the frozen design and matched-capacity RF/ExtraTrees candidates.

### 4. Extract trial order and run prequential evaluation

```bash
paired-eval extract-trial-order \
  --archive /path/to/ia_Paragraph_ordinary.csv.zip \
  --metadata /path/to/bundle/metadata/OneStop_RC.csv \
  --output inputs/onestop_trial_order.csv

paired-eval prequential \
  --design outputs/design \
  --primary outputs/primary \
  --trial-order inputs/onestop_trial_order.csv \
  --output outputs/prequential
```

The trial-order extractor rejects duplicate participant-local indices. During evaluation, each label becomes available only after its prediction.

### 5. Combine summaries

```bash
paired-eval summarize \
  --primary outputs/primary \
  --representations outputs/representations \
  --prequential outputs/prequential \
  --output outputs/summary.json
```

To recompute a summary from completed cells without refitting, add `--summarize-only` to `primary` or `representations`.

## Release verification

The repository archives the exact reported values and project figures. Verify their checksums and structural agreement with:

```bash
paired-eval verify-release
```

Expected top-level result:

```json
{
  "status": "pass"
}
```

The continuous-integration workflow installs the package, runs the synthetic unit tests, and performs the same release verification on every push and pull request.

## Determinism and restart behavior

Random seeds for fold assignment, cells, and bootstraps are archived in [`../results/paper_results.json`](../results/paper_results.json). Each completed cell records hashes for its assignment table, candidate specification, and feature matrix. A cell is reused only if its context and output checksums validate; otherwise it must be recomputed.

## Troubleshooting

- **Undefined AUROC:** verify that each requested evaluation subset contains both outcome classes. The design command checks this for the expected outer and inner regimes.
- **Feature alignment error:** confirm that every metadata `unique_trial_id` occurs exactly once in each applicable feature source.
- **SB-SAT scanpath mismatch:** keep all question responses associated with the same reader–story scanpath in one fold.
- **Oversubscribed CPU:** prefer multiple workers with `--threads 1`; increase per-model threads only when worker count is small.
