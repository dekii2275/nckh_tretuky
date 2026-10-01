# Per-class recall gap

All recall values are percentages; differences are percentage points. Paper reference diagonals are supplied by the user. No cause is inferred.

Ranking uses the larger of the S1/S3 clip recall gaps. S1/S3 have 275 validation clips, R0 has 283; supports are included in the CSV. Small-support classes need cautious interpretation.

| class_name | paper_recall | s1_clip_recall | s3_clip_recall | paper_minus_s1_clip | paper_minus_s3_clip | support_s1_clip |
|---|---|---|---|---|---|---|
| maracas_forward_shaking_mfs | 97.170 | 64.000 | 56.000 | 33.170 | 41.170 | 25 |
| chest_expansion_ce | 98.670 | 75.862 | 62.069 | 22.808 | 36.601 | 29 |
| arm_swing_as | 96.550 | 66.667 | 72.727 | 29.883 | 23.823 | 33 |
| body_swing_bs | 92.880 | 63.333 | 76.667 | 29.547 | 16.213 | 30 |
| drumming_dr | 98.510 | 78.333 | 73.333 | 20.177 | 25.177 | 60 |
| tree_pose_tr | 94.060 | 75.000 | 75.000 | 19.060 | 19.060 | 4 |
| twist_pose_tw | 99.780 | 83.333 | 83.333 | 16.447 | 16.447 | 6 |
| squat_sq | 96.010 | 82.353 | 94.118 | 13.657 | 1.892 | 17 |
| maracas_shaking_ms | 81.100 | 81.579 | 78.947 | -0.479 | 2.153 | 38 |
| sing_and_clap_sac | 76.330 | 75.758 | 75.758 | 0.572 | 0.572 | 33 |

## s1_clip

Largest weighted contributions: drumming_dr, arm_swing_as, body_swing_bs, maracas_forward_shaking_mfs, chest_expansion_ce.
Within 5 percentage points of paper recall: maracas_shaking_ms, sing_and_clap_sac.
Recall shortfall at least 20 points: arm_swing_as, body_swing_bs, chest_expansion_ce, drumming_dr, maracas_forward_shaking_mfs.

| class_name | support | recall_gap_pp | weighted_gap_contribution_pp | actual_errors |
|---|---|---|---|---|
| drumming_dr | 60 | 20.177 | 4.402 | 13 |
| arm_swing_as | 33 | 29.883 | 3.586 | 11 |
| body_swing_bs | 30 | 29.547 | 3.223 | 11 |
| maracas_forward_shaking_mfs | 25 | 33.170 | 3.015 | 9 |
| chest_expansion_ce | 29 | 22.808 | 2.405 | 7 |
| squat_sq | 17 | 13.657 | 0.844 | 3 |
| twist_pose_tw | 6 | 16.447 | 0.359 | 1 |
| tree_pose_tr | 4 | 19.060 | 0.277 | 1 |
| sing_and_clap_sac | 33 | 0.572 | 0.069 | 8 |
| maracas_shaking_ms | 38 | -0.479 | -0.066 | 7 |

## s3_clip

Largest weighted contributions: drumming_dr, chest_expansion_ce, maracas_forward_shaking_mfs, arm_swing_as, body_swing_bs.
Within 5 percentage points of paper recall: maracas_shaking_ms, sing_and_clap_sac, squat_sq.
Recall shortfall at least 20 points: arm_swing_as, chest_expansion_ce, drumming_dr, maracas_forward_shaking_mfs.

| class_name | support | recall_gap_pp | weighted_gap_contribution_pp | actual_errors |
|---|---|---|---|---|
| drumming_dr | 60 | 25.177 | 5.493 | 16 |
| chest_expansion_ce | 29 | 36.601 | 3.860 | 11 |
| maracas_forward_shaking_mfs | 25 | 41.170 | 3.743 | 11 |
| arm_swing_as | 33 | 23.823 | 2.859 | 9 |
| body_swing_bs | 30 | 16.213 | 1.769 | 7 |
| twist_pose_tw | 6 | 16.447 | 0.359 | 1 |
| maracas_shaking_ms | 38 | 2.153 | 0.297 | 8 |
| tree_pose_tr | 4 | 19.060 | 0.277 | 1 |
| squat_sq | 17 | 1.892 | 0.117 | 1 |
| sing_and_clap_sac | 33 | 0.572 | 0.069 | 8 |

Weighted contributions apply the paper recalls to our validation class distribution. Their sum is the gap to that hypothetical weighted recall, not a decomposition of the published 94.84% accuracy: the paper's class supports and evaluation population are not given here.
