# Results and interpretation

This page complements the concise manuscript tables with the exact values archived in [`paper_results.json`](../results/paper_results.json).

## Primary paired comparison

| Dataset | Regime | Base AUROC | Augmented AUROC | Gain | 95% interval |
|---|---|---:|---:|---:|---:|
| OneStop | UT | 0.5609 | 0.6196 | +0.0587 | [0.030, 0.090] |
| OneStop | UR | 0.6212 | 0.6990 | +0.0778 | [0.055, 0.102] |
| SB-SAT | UT | 0.5637 | 0.5739 | +0.0102 | Not estimable |
| SB-SAT | UR | 0.5419 | 0.6954 | +0.1535 | [0.072, 0.226] |

All four point estimates are positive. Conditional bootstrap intervals exclude zero for both OneStop regimes and SB-SAT UR. SB-SAT contains four stories, so too few text-resampling replicates retain both outcome classes to estimate the UT interval under the prespecified valid-replicate rule.

## Controlled representation analysis

### OneStop

| Input | Base UR | Augmented UR | Question gain | Reader gain | Question-gain reduction |
|---|---:|---:|---:|---:|---:|
| Joint baseline | 0.6242 | 0.7009 | +7.68 pp | +5.99 pp | — |
| + question hash | 0.6843 | 0.7021 | +1.78 pp | +5.97 pp | 5.90 pp |
| + option hash | 0.7020 | 0.7053 | +0.34 pp | +6.01 pp | 7.34 pp |

The paired interval for the reduction from the joint baseline to the final arm is **[5.19, 9.66] pp**. The reader-level contrast remains stable while question-varying inputs raise base UR performance and leave little additional gain for the question offset.

### SB-SAT

| Input | Base UR | Augmented UR | Question gain | Reader gain | Question-gain reduction |
|---|---:|---:|---:|---:|---:|
| Scanpath | 0.5354 | 0.6915 | +15.61 pp | +0.45 pp | — |
| + lexical descriptors | 0.6824 | 0.6873 | +0.49 pp | +3.24 pp | 15.12 pp |

The paired interval for the reduction is **[7.54, 21.93] pp**. Because one SB-SAT scanpath is shared by five question trials, adding passage- and question-level descriptors lets the current-trial model distinguish inputs that the scanpath representation alone cannot.

The controlled analysis identifies a representational explanation for most of the question-level gain. It does not isolate semantic content from question identity: fixed text features can serve both roles when questions recur across readers.

## Prequential reader adaptation

The analysis includes 7,918 OneStop trials from 180 readers after requiring ten earlier responses.

| Penalty | Pooled AUROC | Pooled gain (95% interval) | Mean within-reader AUROC | Within-reader gain (95% interval) |
|---:|---:|---:|---:|---:|
| Base | 0.5265 | — | 0.5227 | — |
| 0.1 | 0.5792 | +0.0524 [0.0329, 0.0725] | 0.4604 | −0.0623 [−0.0726, −0.0523] |
| 1 | 0.5710 | +0.0443 [0.0278, 0.0612] | 0.4753 | −0.0473 [−0.0558, −0.0387] |
| 10 | 0.5490 | +0.0223 [0.0144, 0.0305] | 0.5051 | −0.0175 [−0.0237, −0.0115] |
| 100 | 0.5312 | +0.0047 [0.0031, 0.0064] | 0.5198 | −0.0029 [−0.0043, −0.0016] |

The direction is consistent across all prespecified penalties: reader history improves pooled discrimination but not within-reader ordering. This pattern indicates that the correction primarily shifts score levels between readers.

## Robustness checks

The paired gain remains positive in all nine combinations of SB-SAT reader partition and model family and in all ten official OneStop UT/UR fold comparisons. Alternative out-of-fold constructions and separate selection objectives differ from the primary construction by at most 0.5 percentage points; their intervals include zero. These reported checks are encoded in the `robustness` section of `paper_results.json`.

## Scope of the evidence

The results estimate the contribution of the specified regularized offsets conditional on the evaluated tree predictors. The nested design controls model and penalty selection within each outer-training partition. Bootstrap intervals quantify sensitivity to sampled readers and texts while conditioning on saved predictions.
