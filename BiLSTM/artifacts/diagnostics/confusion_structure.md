# Confusion structure

Paper off-diagonal reference is available for only three pairs. Similarity on those pairs cannot establish similarity of the full confusion matrix or exact reproduction.

| class_true | class_pred | paper_percent | r0_percent | s1_window_percent | s1_clip_percent | s3_window_percent | s3_clip_percent |
|---|---|---|---|---|---|---|---|
| body_swing_bs | chest_expansion_ce | 5.360 | 16.667 | 12.632 | 10.000 | 8.657 | 6.667 |
| maracas_shaking_ms | maracas_forward_shaking_mfs | 9.420 | 37.500 | 2.804 | 0.000 | 6.612 | 7.895 |
| sing_and_clap_sac | drumming_dr | 20.390 | 22.857 | 15.625 | 18.182 | 17.768 | 18.182 |

## s1_window: top 20 nonzero pairs by error count

| rank_by_error_count | class_true | class_pred | count | row_support | percent | paper_percent |
|---|---|---|---|---|---|---|
| 1 | sing_and_clap_sac | drumming_dr | 20 | 128 | 15.625 | 20.390 |
| 2 | drumming_dr | sing_and_clap_sac | 14 | 157 | 8.917 | N/A |
| 3 | maracas_shaking_ms | drumming_dr | 13 | 107 | 12.150 | N/A |
| 4 | body_swing_bs | arm_swing_as | 12 | 95 | 12.632 | N/A |
| 5 | body_swing_bs | chest_expansion_ce | 12 | 95 | 12.632 | 5.360 |
| 6 | maracas_forward_shaking_mfs | maracas_shaking_ms | 11 | 73 | 15.068 | N/A |
| 7 | arm_swing_as | chest_expansion_ce | 9 | 71 | 12.676 | N/A |
| 8 | chest_expansion_ce | body_swing_bs | 9 | 80 | 11.250 | N/A |
| 9 | maracas_forward_shaking_mfs | sing_and_clap_sac | 8 | 73 | 10.959 | N/A |
| 10 | maracas_forward_shaking_mfs | arm_swing_as | 6 | 73 | 8.219 | N/A |
| 11 | arm_swing_as | maracas_forward_shaking_mfs | 5 | 71 | 7.042 | N/A |
| 12 | maracas_shaking_ms | tree_pose_tr | 5 | 107 | 4.673 | N/A |
| 13 | body_swing_bs | drumming_dr | 4 | 95 | 4.211 | N/A |
| 14 | sing_and_clap_sac | body_swing_bs | 4 | 128 | 3.125 | N/A |
| 15 | sing_and_clap_sac | chest_expansion_ce | 4 | 128 | 3.125 | N/A |
| 16 | drumming_dr | body_swing_bs | 4 | 157 | 2.548 | N/A |
| 17 | arm_swing_as | drumming_dr | 3 | 71 | 4.225 | N/A |
| 18 | chest_expansion_ce | arm_swing_as | 3 | 80 | 3.750 | N/A |
| 19 | chest_expansion_ce | drumming_dr | 3 | 80 | 3.750 | N/A |
| 20 | chest_expansion_ce | twist_pose_tw | 3 | 80 | 3.750 | N/A |

## s1_clip: top 20 nonzero pairs by error count

| rank_by_error_count | class_true | class_pred | count | row_support | percent | paper_percent |
|---|---|---|---|---|---|---|
| 1 | drumming_dr | sing_and_clap_sac | 8 | 60 | 13.333 | N/A |
| 2 | sing_and_clap_sac | drumming_dr | 6 | 33 | 18.182 | 20.390 |
| 3 | arm_swing_as | chest_expansion_ce | 5 | 33 | 15.152 | N/A |
| 4 | maracas_forward_shaking_mfs | maracas_shaking_ms | 4 | 25 | 16.000 | N/A |
| 5 | body_swing_bs | arm_swing_as | 4 | 30 | 13.333 | N/A |
| 6 | maracas_shaking_ms | drumming_dr | 4 | 38 | 10.526 | N/A |
| 7 | body_swing_bs | chest_expansion_ce | 3 | 30 | 10.000 | 5.360 |
| 8 | maracas_forward_shaking_mfs | arm_swing_as | 2 | 25 | 8.000 | N/A |
| 9 | maracas_forward_shaking_mfs | sing_and_clap_sac | 2 | 25 | 8.000 | N/A |
| 10 | chest_expansion_ce | body_swing_bs | 2 | 29 | 6.897 | N/A |
| 11 | chest_expansion_ce | twist_pose_tw | 2 | 29 | 6.897 | N/A |
| 12 | body_swing_bs | drumming_dr | 2 | 30 | 6.667 | N/A |
| 13 | arm_swing_as | maracas_forward_shaking_mfs | 2 | 33 | 6.061 | N/A |
| 14 | arm_swing_as | tree_pose_tr | 2 | 33 | 6.061 | N/A |
| 15 | drumming_dr | tree_pose_tr | 2 | 60 | 3.333 | N/A |
| 16 | tree_pose_tr | squat_sq | 1 | 4 | 25.000 | N/A |
| 17 | twist_pose_tw | drumming_dr | 1 | 6 | 16.667 | N/A |
| 18 | squat_sq | body_swing_bs | 1 | 17 | 5.882 | N/A |
| 19 | squat_sq | drumming_dr | 1 | 17 | 5.882 | N/A |
| 20 | squat_sq | tree_pose_tr | 1 | 17 | 5.882 | N/A |

## s3_window: top 20 nonzero pairs by error count

| rank_by_error_count | class_true | class_pred | count | row_support | percent | paper_percent |
|---|---|---|---|---|---|---|
| 1 | sing_and_clap_sac | drumming_dr | 78 | 439 | 17.768 | 20.390 |
| 2 | drumming_dr | sing_and_clap_sac | 59 | 526 | 11.217 | N/A |
| 3 | maracas_forward_shaking_mfs | maracas_shaking_ms | 55 | 251 | 21.912 | N/A |
| 4 | maracas_shaking_ms | drumming_dr | 39 | 363 | 10.744 | N/A |
| 5 | maracas_forward_shaking_mfs | arm_swing_as | 38 | 251 | 15.139 | N/A |
| 6 | body_swing_bs | arm_swing_as | 33 | 335 | 9.851 | N/A |
| 7 | arm_swing_as | chest_expansion_ce | 30 | 226 | 13.274 | N/A |
| 8 | chest_expansion_ce | arm_swing_as | 29 | 270 | 10.741 | N/A |
| 9 | body_swing_bs | chest_expansion_ce | 29 | 335 | 8.657 | 5.360 |
| 10 | chest_expansion_ce | body_swing_bs | 27 | 270 | 10.000 | N/A |
| 11 | maracas_shaking_ms | maracas_forward_shaking_mfs | 24 | 363 | 6.612 | 9.420 |
| 12 | drumming_dr | body_swing_bs | 23 | 526 | 4.373 | N/A |
| 13 | sing_and_clap_sac | body_swing_bs | 20 | 439 | 4.556 | N/A |
| 14 | maracas_forward_shaking_mfs | sing_and_clap_sac | 16 | 251 | 6.375 | N/A |
| 15 | chest_expansion_ce | drumming_dr | 16 | 270 | 5.926 | N/A |
| 16 | maracas_shaking_ms | body_swing_bs | 16 | 363 | 4.408 | N/A |
| 17 | maracas_forward_shaking_mfs | body_swing_bs | 14 | 251 | 5.578 | N/A |
| 18 | body_swing_bs | maracas_forward_shaking_mfs | 14 | 335 | 4.179 | N/A |
| 19 | arm_swing_as | drumming_dr | 12 | 226 | 5.310 | N/A |
| 20 | arm_swing_as | maracas_forward_shaking_mfs | 12 | 226 | 5.310 | N/A |

## s3_clip: top 20 nonzero pairs by error count

| rank_by_error_count | class_true | class_pred | count | row_support | percent | paper_percent |
|---|---|---|---|---|---|---|
| 1 | drumming_dr | sing_and_clap_sac | 10 | 60 | 16.667 | N/A |
| 2 | sing_and_clap_sac | drumming_dr | 6 | 33 | 18.182 | 20.390 |
| 3 | maracas_forward_shaking_mfs | maracas_shaking_ms | 5 | 25 | 20.000 | N/A |
| 4 | chest_expansion_ce | arm_swing_as | 5 | 29 | 17.241 | N/A |
| 5 | arm_swing_as | chest_expansion_ce | 4 | 33 | 12.121 | N/A |
| 6 | maracas_forward_shaking_mfs | arm_swing_as | 3 | 25 | 12.000 | N/A |
| 7 | chest_expansion_ce | body_swing_bs | 3 | 29 | 10.345 | N/A |
| 8 | body_swing_bs | arm_swing_as | 3 | 30 | 10.000 | N/A |
| 9 | maracas_shaking_ms | drumming_dr | 3 | 38 | 7.895 | N/A |
| 10 | maracas_shaking_ms | maracas_forward_shaking_mfs | 3 | 38 | 7.895 | 9.420 |
| 11 | maracas_forward_shaking_mfs | body_swing_bs | 2 | 25 | 8.000 | N/A |
| 12 | chest_expansion_ce | drumming_dr | 2 | 29 | 6.897 | N/A |
| 13 | body_swing_bs | chest_expansion_ce | 2 | 30 | 6.667 | 5.360 |
| 14 | arm_swing_as | squat_sq | 2 | 33 | 6.061 | N/A |
| 15 | drumming_dr | body_swing_bs | 2 | 60 | 3.333 | N/A |
| 16 | drumming_dr | maracas_shaking_ms | 2 | 60 | 3.333 | N/A |
| 17 | tree_pose_tr | squat_sq | 1 | 4 | 25.000 | N/A |
| 18 | twist_pose_tw | drumming_dr | 1 | 6 | 16.667 | N/A |
| 19 | squat_sq | tree_pose_tr | 1 | 17 | 5.882 | N/A |
| 20 | maracas_forward_shaking_mfs | sing_and_clap_sac | 1 | 25 | 4.000 | N/A |

S3's three supplied pairs are close in absolute percentage, but other large reproduction errors must be examined separately. Missing paper off-diagonal entries are marked N/A, not treated as zero.
