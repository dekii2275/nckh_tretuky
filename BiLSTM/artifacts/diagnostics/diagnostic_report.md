# Reproduction diagnostics after R0/S1/S2/S3

The published architecture remains frozen at 860362 trainable parameters. No source model, preprocessing implementation, training configuration or frozen split was changed, and no model was trained in this phase. R0/S1/S2/S3 artifacts are protected by SHA256.

## A. Per-class recall gap

Paper diagonal recalls below were supplied by the user. Recall is %, gap is percentage points; the table is ranked by the larger S1/S3 gap. R0 and window-level recalls/supports are also in per_class_gap.csv.

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

The largest raw recall deficits are Maracas Forward Shaking, Chest Expansion, Arm Swing, Body Swing and Drumming. Drumming contributes the most support-weighted deficit in both S1 and S3 (4.40 and 5.49 points against paper recalls applied to our validation distribution). This is not a decomposition of the published 94.84% accuracy, whose class supports are unknown here.

Maracas Shaking and Sing and Clap are within about 2.2 recall points of the supplied diagonal in both protocols; S3 Squat is within 1.90 points. No cause follows from per-class recall alone.

## B. Confusion structure

| class_true | class_pred | paper_percent | r0_percent | s1_clip_percent | s3_clip_percent |
|---|---|---|---|---|---|
| body_swing_bs | chest_expansion_ce | 5.360 | 16.667 | 10.000 | 6.667 |
| maracas_shaking_ms | maracas_forward_shaking_mfs | 9.420 | 37.500 | 0.000 | 7.895 |
| sing_and_clap_sac | drumming_dr | 20.390 | 22.857 | 18.182 | 18.182 |

Top five s1_clip confusions by number of errors:

| class_true | class_pred | count | row_support | percent | paper_percent |
|---|---|---|---|---|---|
| drumming_dr | sing_and_clap_sac | 8 | 60 | 13.333 |  |
| sing_and_clap_sac | drumming_dr | 6 | 33 | 18.182 | 20.39 |
| arm_swing_as | chest_expansion_ce | 5 | 33 | 15.152 |  |
| maracas_forward_shaking_mfs | maracas_shaking_ms | 4 | 25 | 16.000 |  |
| body_swing_bs | arm_swing_as | 4 | 30 | 13.333 |  |

Top five s3_clip confusions by number of errors:

| class_true | class_pred | count | row_support | percent | paper_percent |
|---|---|---|---|---|---|
| drumming_dr | sing_and_clap_sac | 10 | 60 | 16.667 |  |
| sing_and_clap_sac | drumming_dr | 6 | 33 | 18.182 | 20.39 |
| maracas_forward_shaking_mfs | maracas_shaking_ms | 5 | 25 | 20.000 |  |
| chest_expansion_ce | arm_swing_as | 5 | 29 | 17.241 |  |
| arm_swing_as | chest_expansion_ce | 4 | 33 | 12.121 |  |

All top-20 nonzero pairs at window and clip levels are in top20_confusion_pairs.csv. Only three off-diagonal paper entries are supplied; structural similarity of the full paper matrix is therefore not established. Similar selected entries do not prove reproduction.

## C. Human tracking audit

Reviewed 0/120. A=0, B=0, C=0, D=0. All A/B/C/D rates are unavailable, not zero error rates. Current status: awaiting_human_review.

The existing 120-clip raw-frame set is reused. index.html and each clip page support A/B/C/D, secondary flags, CSV export/import, candidate indices, selected index, bounding boxes, MidHip and confidence. Previous browser label storage is not converted into human A/B/C/D labels automatically.

tracking_vs_accuracy.csv will compare saved S1 validation accuracy/F1 for A/B/C/D and B+C once labels exist. It excludes training clips and short clips without S1 validation predictions. Rates by activity/session/subject/max-person-count are in tracking_label_breakdown.csv. The deliberately stratified review set is not an unbiased whole-dataset prevalence sample.

## D. Child-size assumption

Status: awaiting_human_role_pairs; 0 confidently human-identified frame pairs. Percent child-smaller/larger/ambiguous is unavailable. manual_person_pairs.csv and the UI let a reviewer identify child/therapist raw indices at the same frame. analyze_child_size.py then measures original bbox area and skeleton height, with a predeclared +/-5% ambiguous band and clip-weighted median ratios.

A/B/C/D alone does not identify both people's boxes. No role is inferred from size, appearance or the external ASD_Label. The existing smaller-person prior remains unchanged. No oracle subset test or candidate tracker is prepared/run without the required human evidence.

## E. Local tracking assets

Read-only inventory found 3276 MediaPipe CSVs, 3267 optical-flow MP4s and 642636 ROMP NPZ frames. Stored ASD_Label file counts: {'0': 1315, '1': 1258, '6': 1, '2': 697, '3': 5}. Values beyond 0/1 have unclear semantics in the inspected local code.

Filename candidates cover 1069/1374 frozen clips, but all MediaPipe CSVs have 179 data rows and none of the matched CSVs has an equal raw OpenPose frame count. Crop-to-original coordinates and frame/person mappings are unverified. README/code describe YOLO/DeepSORT and person labels; feature outputs and existing labels are not a validated child-ID mapping. See external_tracking_assets.md for evidence and limitations.

Validated external role metadata could eventually select original 2D OpenPose persons while preserving 60×50 input. Feeding MediaPipe 3D, ROMP or optical-flow features would change the input protocol and is not proposed.

## F. Actual preprocessing usage

All 1374 frozen clips are audited before temporal sampling. Frame/joint rates are pooled by split, not means of clip percentages. The implementation is unchanged.

| split | frames | missing_MidHip_% | missing_Lshoulder_% | missing_Rshoulder_% | invalid_scale/torso_fallback_% | filled_missing_joints_% | bounded_interpolation_% | edge_fill_% | frames_gt25%missing_% | severe_clips_% |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 201384 | 1.657 | 0.943 | 0.847 | 1.408 | 10.704 | 6.639 | 4.065 | 22.136 | 26.856 |
| val | 53290 | 1.162 | 0.345 | 0.368 | 0.903 | 9.305 | 5.909 | 3.396 | 14.414 | 16.608 |

Severe missingness is a predeclared diagnostic flag: more than 25% of clip frames have more than 25% invalid joints. Filled joints includes bounded interpolation and leading/trailing extension, but excludes never-observed joints. Torso fallback includes its existing span/1.0 secondary safeguards. Per-class tables, original scale counters and full run-length distributions are saved separately.

## G. Validation error factors

S1 mean-logit clip predictions: 204 correct / 71 incorrect on 275 retained validation clips. The error table preserves all 283 validation clips, with prediction fields blank for the eight dropped short clips.

| classification | clips | missing_joints_% | torso_fallback_% | mean_keypoint_confidence | mean_people | prediction_confidence | logit_margin |
|---|---|---|---|---|---|---|---|
| correct | 204 | 8.316 | 0.503 | 0.663 | 3.442 | 0.966 | 7.522 |
| incorrect | 71 | 20.608 | 2.114 | 0.547 | 3.680 | 0.873 | 4.511 |

Errors are descriptively associated with higher missingness/fallback frequency, lower keypoint confidence and a slightly larger mean raw candidate count. Prediction confidence is uncalibrated softmax of mean logits. Tracking ambiguity is unmeasured until human labels exist; raw-index changes cannot substitute for identity-switch labels. Class composition and other confounders remain.

## Current split limitation

Tree Pose: 4 validation clips (3.10% of that class); Twist Pose: 6 (5.00%). Both use a single validation subject/session, 41023/D8. These small supports make recalls unstable. The current split was not modified. No future seed candidates were generated because the human tracking-quality gate has not passed. Future seeds remain predeclared as 0,1,2,3,4 with S1, never selected by accuracy.

## H. Exactly one primary next step

1. improve child selection

Complete the prepared human child/person identity audit first; only then decide whether the tracker needs correction. Zero human labels cannot establish that tracking is good or bad. Do not modify the tracker yet.

The automatic diagnostics and review tooling are complete. Human identity review and human role-pair measurements remain pending, so the primary cause of the remaining paper gap is not yet determined.

## Files and rerun commands

Open `../tracking_review/index.html`. Export labels/pairs, replace `../tracking_review/manual_labels.csv` and `manual_person_pairs.csv`, then run from BiLSTM:

```bash
python scripts/analyze_tracking_labels.py
python scripts/analyze_child_size.py
python scripts/report_diagnostics.py
```

Preservation check: 88 protected experiment/config/source files unchanged. No full retraining, architecture tuning, tracker replacement, oracle test or seed sweep occurred.
