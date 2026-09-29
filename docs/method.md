# Method

This document expands the evaluation design summarized in the manuscript. It is intended as a technical reference for reproducing or adapting the pipeline.

## Evaluation target

For reader `u` and question `i`, the binary outcome is response correctness and `x[u,i]` contains only features of the current trial. A base model produces `q[u,i] = F(x[u,i])`. The paired contrast asks how much predictive performance changes when a regularized correction estimated from **other response labels** is added to this fixed base score.

The target is therefore conditional on the evaluated predictor family and offset form. It is not a psychometric estimate of reader ability or question difficulty.

## Information regimes

Questions are nested in complete text groups: a OneStop article or an SB-SAT story.

| Regime | Reader status | Text status | Added correction |
|---|---|---|---|
| UT | Seen | Unseen | Reader offset |
| UR | Unseen | Seen | Question offset |
| UB | Unseen | Unseen | None |

UT can use labeled responses from the same reader to other texts. UR can use labeled responses from other readers to the same question. UB provides neither source.

## Cross-fitted offsets

Offsets are fitted against out-of-fold (OOF) base scores rather than scores produced by a model trained on the same labels.

1. Text-held-out cross-fitting produces the base-score view used for reader offsets.
2. Reader-held-out cross-fitting produces the base-score view used for question offsets.
3. Each path uses three folds inside the current outer-training set.
4. Responses that share an SB-SAT scanpath remain in the same fold.

Let `s(q)` be a clipped logit. A shared intercept and L2-regularized reader and question offsets minimize the sum of the two regime-matched logistic losses. Each response enters once through the text-held-out view and once through the reader-held-out view. The reader and question penalties are selected from `{0.1, 1, 10, 100}`.

After selection, the base predictor is refitted on the complete outer-training partition. The paired predictions are:

```text
UT base: q              UT augmented: sigmoid(logit(q) + intercept + reader offset)
UR base: q              UR augmented: sigmoid(logit(q) + intercept + question offset)
UB base: q              UB augmented: sigmoid(logit(q) + intercept)
```

The intercept cannot alter AUROC ranking. A reader offset is constant within a reader and a question offset is constant within a question; they can change between-group comparisons but not rankings within the corrected group.

## Nested reader-by-text evaluation

Reader and text identifiers are assigned to folds without consulting labels. OneStop uses five folds per axis and SB-SAT uses four. For outer cell `(r, t)`:

- training excludes every row whose reader is in fold `r` **or** whose text is in fold `t`;
- UB test rows are in both held-out folds;
- UT test rows have a held-out text and a reader represented in training;
- UR test rows have a held-out reader and a text represented in training.

The inner evaluation uses a three-by-three reader–text Cartesian design and the same grouping rules. All learned preprocessing is fitted only on the relevant training rows.

Selection has two stages:

1. the equal-weight mean of base AUROC across UT, UR, and UB selects the feature view and tree candidate;
2. with that base predictor fixed, the equal-weight mean of augmented AUROC selects the offset penalties.

Mean Brier score and fixed candidate order provide deterministic tie-breaking. Outer-test performance is not used by either selection stage.

## Predictors and representations

The primary search evaluates Random Forest and ExtraTrees candidates with prespecified depth, leaf-size, feature-fraction, and class-weight configurations. Each standard configuration uses 256 trees; an additional shallow balanced Random Forest uses 1,000 trees. Across feature views, the primary grids contain 39 OneStop and 26 SB-SAT candidates.

The controlled representation analysis holds capacity constant. Each arm uses the same two candidates: Random Forest and ExtraTrees with 32 trees, depth 12, minimum leaf size 5, and feature fraction 0.5.

| Dataset | Baseline representation | Added representation |
|---|---|---|
| OneStop | 15 text + 525 gaze descriptors | 64-dimensional question hash, then 64-dimensional option hash |
| SB-SAT | 142-dimensional scanpath vector | 33 passage/question lexical descriptors |

The question and option hashes use normalized word unigram and bigram term frequencies. SB-SAT lexical descriptors summarize token count, type–token ratio, token length, word-frequency surprisal, and question–passage vocabulary overlap. No representation contains response-label aggregates.

## Prequential reader evaluation

The retrospective UT comparison can use all other labeled responses from a represented reader. The prequential analysis instead orders OneStop trials by participant-local trial index. For each trial:

1. a frozen predictor supplies a base score from a model that excluded the trial's reader and text;
2. a reader correction is fitted using only responses with smaller trial indices;
3. the current response is predicted;
4. its label is then added to the reader history.

Evaluation begins after ten earlier responses. All four penalties are reported without selecting among them. Pooled AUROC measures discrimination over all eligible trials, while mean within-reader AUROC equally weights readers with both outcomes.

## Metrics and uncertainty

The principal endpoint is the paired change in AUROC on identical saved observations. A 2,000-replicate reader-by-text bootstrap independently resamples reader and text identifiers, then applies the product of their sampling multiplicities as observation weights. Representation contrasts reuse the same multiplicities in every arm.

These intervals condition on the fitted predictions. They are reported only when at least 1,000 replicates retain a defined AUROC. The prequential analysis uses a 5,000-replicate reader-cluster bootstrap that preserves each sampled reader's trial sequence.

Exact reported values and random seeds are in [`../results/paper_results.json`](../results/paper_results.json).
