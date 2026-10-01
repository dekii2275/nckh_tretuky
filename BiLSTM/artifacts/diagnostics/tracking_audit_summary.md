# Human tracking-label audit

Status: awaiting_human_review. Reviewed 0/120; unreviewed 120.

A=correct child; B=wrong person; C=starts correctly then switches; D=uncertain. Blank is unreviewed, not D. Rates use reviewed clips including D in the denominator. No labels are inferred from model predictions, size or external binary annotations.

| reviewed | unreviewed | A_count | B_count | C_count | D_count | correct_child_rate | wrong_person_rate | identity_switch_rate | uncertain_rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 120 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |

## Saved S1 classification by human group

Only held-out validation clips with saved S1 predictions are evaluated; training clips and dropped short clips are not assigned fabricated scores. Macro-F1 is over all 10 frozen classes, so class coverage matters.

| review_label | reviewed_clips_all_splits | s1_validation_clips | represented_classes | accuracy | macro_f1 |
|---|---|---|---|---|---|
| A | 0 | 0 | 0 | N/A | N/A |
| B | 0 | 0 | 0 | N/A | N/A |
| C | 0 | 0 | 0 | N/A | N/A |
| D | 0 | 0 | 0 | N/A | N/A |
| B+C | 0 | 0 | 0 | N/A | N/A |

Interpretation: manual_review_required. Difference A minus B+C accuracy: None.

Breakdowns by activity/session/subject and max-person-count category are saved in tracking_label_breakdown.csv. The review set deliberately includes reordering cases; any association is descriptive and cannot establish causality.

Split-sensitivity preparation gate passed: False. No split candidates or training have been run. Do not infer tracker quality from zero human labels.
