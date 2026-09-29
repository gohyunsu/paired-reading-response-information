# Data preparation

The experiments use the ordinary-reading **Gathering** subset of [OneStop](https://osf.io/2prdq/) and the [SB-SAT](https://github.com/ahnchive/SB-SAT) reading-fixation dataset. Dataset files are not duplicated in this repository. Obtain them from their original releases and retain their terms of use.

The command-line interface expects a prepared feature bundle with this layout:

```text
bundle/
├── metadata/
│   ├── OneStop_RC.csv
│   └── SBSAT_RC.csv
├── onestop/
│   ├── text_features.feather
│   └── gaze_features.feather
└── sbsat/
    ├── values.npy
    ├── offsets.npy
    └── sequence_metadata.csv
```

## Metadata tables

Both metadata CSV files require the following columns:

| Column | Meaning |
|---|---|
| `unique_trial_id` | Unique response identifier |
| `participant_id` | Reader identifier |
| `unique_paragraph_id` | OneStop paragraph-version identifier or SB-SAT story identifier |
| `question_id` | Question number within the text |
| `item_key` | Question identifier shared across readers |
| `label` | Binary response correctness |

The implementation constructs `scan_group` as `participant_id::unique_paragraph_id`. A complete text is the fold unit: a OneStop article for OneStop and a story for SB-SAT.

## OneStop features

`text_features.feather` must contain one row per `unique_trial_id`, 15 columns prefixed with `text_`, and the identifier column. `gaze_features.feather` must contain the same trials, a `label` column, and 525 gaze/fixation-aligned linguistic descriptors. The loader aligns both matrices to the metadata table by `unique_trial_id` and rejects duplicate or missing identifiers.

The representation analysis additionally requires the public OneStop question file:

```text
onestop_qa.json
```

Its top-level `data` list contains articles, paragraphs, questions, and answer strings. Question and answer-option hashes are 64-dimensional unsigned unigram/bigram term-frequency vectors with L2 normalization. Option strings are normalized separately and sorted before concatenation; option position is not encoded.

## SB-SAT features

The sequence arrays use a compact ragged representation:

- `values.npy`: all fixation rows concatenated, with the six released channels in the first six columns;
- `offsets.npy`: length `n_trials + 1`, giving the half-open slice for each sequence;
- `sequence_metadata.csv`: one row per sequence with `unique_trial_id` in array order.

Repeated question responses sharing one reader–story scanpath therefore receive the same 142-dimensional scanpath descriptor vector.

The representation analysis also requires `combined_stimulus.csv`, containing the released SB-SAT passage and question text. The 33 lexical descriptors comprise 16 passage descriptors, 16 question-stem descriptors, and one vocabulary-overlap descriptor.

## Trial order for the prequential analysis

Create `onestop_trial_order.csv` from the released ordinary-reading interest-area archive:

```bash
paired-eval extract-trial-order \
  --archive /path/to/ia_Paragraph_ordinary.csv.zip \
  --metadata /path/to/bundle/metadata/OneStop_RC.csv \
  --output /path/to/onestop_trial_order.csv
```

The resulting file has exactly one row per retained OneStop trial:

```text
unique_trial_id,participant_id,trial_index
```

`trial_index` must be unique within each participant. The evaluator sorts each reader’s trials by this field and reveals a correctness label only after producing that trial’s prediction.

## Validation

`paired-eval design` validates row counts, binary labels, group support, and AUROC definability for every outer and inner regime. Each fitted cell records hashes of its assignment table, candidate specification, and feature matrix.
