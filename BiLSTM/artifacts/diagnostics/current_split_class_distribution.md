# Current frozen split composition

Percentages train_percent/val_percent are within each class; class_share_percent is within each split. Counts include short clips; retained clip and S1-window counts are separate.

| class | total | train_count | val_count | val_percent | train_subjects | val_subjects | train_subject_session_groups | val_subject_session_groups |
|---|---|---|---|---|---|---|---|---|
| arm_swing_as | 105 | 72 | 33 | 31.429 | 11 | 7 | 20 | 9 |
| body_swing_bs | 117 | 87 | 30 | 25.641 | 8 | 6 | 16 | 7 |
| chest_expansion_ce | 114 | 84 | 30 | 26.316 | 10 | 6 | 19 | 8 |
| drumming_dr | 343 | 281 | 62 | 18.076 | 12 | 5 | 22 | 6 |
| maracas_forward_shaking_mfs | 101 | 75 | 26 | 25.743 | 11 | 5 | 18 | 7 |
| maracas_shaking_ms | 130 | 90 | 40 | 30.769 | 11 | 5 | 18 | 7 |
| sing_and_clap_sac | 113 | 78 | 35 | 30.973 | 11 | 5 | 17 | 7 |
| squat_sq | 102 | 85 | 17 | 16.667 | 11 | 6 | 21 | 7 |
| tree_pose_tr | 129 | 125 | 4 | 3.101 | 10 | 1 | 12 | 1 |
| twist_pose_tw | 120 | 114 | 6 | 5.000 | 10 | 1 | 15 | 1 |

Uneven class allocation, few held-out subjects and very small validation supports can make class estimates unstable. This is a descriptive composition finding, not proof that the split causes the observed errors.

Future seed set is exactly [0, 1, 2, 3, 4], S1 stride 60. No candidate splits or training configs are generated until human tracking review meets the quality gate. No seed was tested or selected.
