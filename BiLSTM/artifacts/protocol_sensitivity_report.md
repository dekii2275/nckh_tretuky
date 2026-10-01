# Protocol sensitivity study

Using the published architecture and training settings, combined with our current implementation choices for sequence construction, child selection, missing joints, and session split, reproduction R0 achieved 67.84% validation accuracy.

All three strides were predeclared. No tested stride is asserted to be the author's true method.

| Experiment | Train/val clips | Train/val samples | Dropped train/val clips | Best/stop epoch | Window Acc/F1 | Clip Acc/F1 |
|---|---|---|---|---|---|---|
| R0_clip_uniform60 | 1091/283 | 1091/283 | 0/0 | 32/42 | N/A/N/A | 67.8445%/65.7509% |
| S1_window60_stride60 | 1091/283 | 2970/779 | 12/8 | 49/59 | 74.3261%/71.0411% | 74.1818%/71.8218% |
| S2_window60_stride30 | 1091/283 | 5395/1412 | 12/8 | 28/38 | 71.2465%/67.6768% | 74.1818%/71.2842% |
| S3_window60_stride15 | 1091/283 | 9942/2634 | 12/8 | 18/28 | 72.7031%/69.8446% | 73.4545%/72.7935% |

N/A denotes unavailable/not applicable; blank S metrics in CSV denote pending training. R0 samples are clips, not contiguous windows.

Coverage control: on the same 275 retained validation clips, existing R0 predictions have accuracy 67.2727% and macro F1 65.6181%. R0 was not retrained or overwritten.

## Epoch-1 loss diagnostics

| Experiment | Sample-weighted mean | Sum of batch means | Batches |
|---|---|---|---|
| R0_clip_uniform60 | 1.7337603613829853 | None | 35 |
| S1_window60_stride60 | 1.382155430517614 | 128.47248113155365 | 93 |
| S2_window60_stride30 | 1.2109379552049258 | 204.7046855688095 | 169 |
| S3_window60_stride15 | 0.9858160111966975 | 306.4315260350704 | 311 |

R0 batch count is derived from 1091 clips / batch 32. Its batch-loss sum was not logged and is not inferred from the weighted mean.

## Metrics at both levels

| Experiment | Level | Accuracy | Macro precision | Macro recall | Macro F1 | Support |
|---|---|---|---|---|---|---|
| S1_window60_stride60 | window | 74.3261% | 70.1799% | 73.7713% | 71.0411% | 779 |
| S1_window60_stride60 | clip | 74.1818% | 71.4344% | 74.6218% | 71.8218% | 275 |
| S2_window60_stride30 | window | 71.2465% | 66.2356% | 71.9384% | 67.6768% | 1412 |
| S2_window60_stride30 | clip | 74.1818% | 70.2658% | 74.6997% | 71.2842% | 275 |
| S3_window60_stride15 | window | 72.7031% | 69.1703% | 71.4943% | 69.8446% | 2634 |
| S3_window60_stride15 | clip | 73.4545% | 71.7719% | 74.7952% | 72.7935% | 275 |

## Paper confusion diagnostics

| Experiment | Level | Pair | Count/support | Percent | Supplied paper reference |
|---|---|---|---|---|---|
| R0_clip_uniform60 | clip | BS -> CE | 5/30 | 16.666666666666668 | 5.36 |
| R0_clip_uniform60 | clip | MS -> MFS | 15/40 | 37.5 | 9.42 |
| R0_clip_uniform60 | clip | SAC -> DR | 8/35 | 22.857142857142858 | 20.39 |
| S1_window60_stride60 | window | BS -> CE | 12/95 | 12.631578947368421 | 5.36 |
| S1_window60_stride60 | window | MS -> MFS | 3/107 | 2.803738317757009 | 9.42 |
| S1_window60_stride60 | window | SAC -> DR | 20/128 | 15.625 | 20.39 |
| S1_window60_stride60 | clip | BS -> CE | 3/30 | 10.0 | 5.36 |
| S1_window60_stride60 | clip | MS -> MFS | 0/38 | 0.0 | 9.42 |
| S1_window60_stride60 | clip | SAC -> DR | 6/33 | 18.181818181818183 | 20.39 |
| S2_window60_stride30 | window | BS -> CE | 25/177 | 14.124293785310735 | 5.36 |
| S2_window60_stride30 | window | MS -> MFS | 9/198 | 4.545454545454546 | 9.42 |
| S2_window60_stride30 | window | SAC -> DR | 52/233 | 22.317596566523605 | 20.39 |
| S2_window60_stride30 | clip | BS -> CE | 3/30 | 10.0 | 5.36 |
| S2_window60_stride30 | clip | MS -> MFS | 2/38 | 5.2631578947368425 | 9.42 |
| S2_window60_stride30 | clip | SAC -> DR | 6/33 | 18.181818181818183 | 20.39 |
| S3_window60_stride15 | window | BS -> CE | 29/335 | 8.656716417910447 | 5.36 |
| S3_window60_stride15 | window | MS -> MFS | 24/363 | 6.6115702479338845 | 9.42 |
| S3_window60_stride15 | window | SAC -> DR | 78/439 | 17.767653758542142 | 20.39 |
| S3_window60_stride15 | clip | BS -> CE | 2/30 | 6.666666666666667 | 5.36 |
| S3_window60_stride15 | clip | MS -> MFS | 3/38 | 7.894736842105263 | 9.42 |
| S3_window60_stride15 | clip | SAC -> DR | 6/33 | 18.181818181818183 | 20.39 |

Full per-class percentage resolution and observed off-diagonal minima: `window_audit/confusion_granularity.csv`. Larger window row supports allow finer percentage increments; this is diagnostic evidence only and does not prove that the paper used windows.

## Interpretation

Sequence construction is a major sensitivity factor (>5 percentage points in clip accuracy). This does not establish the author's method or exact reproduction. S1_window60_stride60: clip accuracy change +6.34 pp versus full R0, +6.91 pp versus R0 on the same retained clips; remaining gap to 94.84% = 20.66 pp. S2_window60_stride30: clip accuracy change +6.34 pp versus full R0, +6.91 pp versus R0 on the same retained clips; remaining gap to 94.84% = 20.66 pp. S3_window60_stride15: clip accuracy change +5.61 pp versus full R0, +6.18 pp versus R0 on the same retained clips; remaining gap to 94.84% = 21.39 pp. All clip results remain below 80%; sequence construction does not explain the full paper gap. Proceed to manual child/person identity review, retaining the architecture and tracker until review evidence is available.

Paper reference supplied for comparison: validation accuracy 94.84%, macro precision/recall/F1 approximately 94%.

## Remaining assumptions

- The conversion from variable clip length to 60 frames, stride, short-clip policy and tail policy are unreported; S1/S2/S3 are predeclared sensitivity choices.
- R0 uses rounded uniform source indices for long clips and linear temporal interpolation for short clips. Windows use original frame positions; full-clip missing-joint interpolation remains unchanged.
- Frame indices refer to the ordered OpenPose JSON files, zero based and end inclusive; no invented frames are inserted for missing JSON files.
- Child selection uses temporal_nearest and prefers smaller candidates on initialization; it is not manually verified child identity.
- Confidence threshold 0.1, unlimited missing-joint interpolation including edge filling, zero fill before normalization for never-observed joints, no smoothing, shoulder threshold 5 px and torso fallback are implementation choices.
- MidHip centering and shoulder normalization follow the published formulas; behavior when joints/scale are invalid is implementation-specific.
- Frog Pose is excluded because local directories have no usable OpenPose JSON; the paper does not explicitly explain its 10-class exclusion.
- The exact author session split and random seed are unreported. This study reuses the existing 1091/283 subject-session split and seed 42; subjects may occur in both sets at different sessions.
- Attention dropout and internal LSTM dropout are zero; Adam uses existing PyTorch defaults and zero weight decay; no scheduler, augmentation, class weighting, clipping or AMP is added.
- Validation accuracy is the existing early-stopping monitor with strict improvement, patience 10 and a maximum of 100 epochs. Window runs monitor window accuracy; clip aggregation is predeclared mean logits at the selected checkpoint.
- Window samples weight longer clips more heavily; overlapping windows are correlated. Clip evaluation excludes clips shorter than 60 frames, so coverage must accompany comparisons with R0.
- Epoch loss mean is sample-weighted CrossEntropyLoss; batch loss sum is the unweighted sum of batch means. R0 did not log batch sums, which cannot be reconstructed exactly from its weighted epoch mean.
- The paper's loss reduction and confusion-matrix normalization/evaluation unit are unreported. Reference figures here are supplied by the user, not independently verified against the paper.
- R0 is a legacy checkpoint without original training manifest hashes. Its current evaluation and current split are verified, but this cannot prove its historical training inputs; raw JSON contents are not hashed.
- Deterministic seeding retains the existing warn_only behavior and runtime defaults; bitwise reproducibility across hardware/software is not claimed.

## Manual identity review prepared

Exported 120 clips across 10 classes, 32 subjects and days D1/D8/D16. Open `tracking_review/index.html` for all-frame skeleton playback, candidate indices, selected track, bounding boxes, MidHip and confidence. This is a skeleton-only manual review, not child ground truth; RGB may be needed for ambiguous identity.

Raw-candidate coverage: population single-person-only clips = 0; population clips with some one-person frames = 5; selected clips with one-person frames = 3; selected clips with multi-person frames = 120. Days: {'D1': 47, 'D8': 43, 'D16': 30}.

Reordering strata use a documented geometric matching proxy, not verified identity switches. The tracker and model remain unchanged. No human review labels have been assigned automatically.
