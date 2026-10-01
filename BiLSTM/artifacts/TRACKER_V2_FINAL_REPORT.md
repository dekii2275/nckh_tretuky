# Tracker V2 — implementation, audit, and controlled training report

## Executive decision

**Tracker V2: PASS.** On the frozen 1,374-clip manifest (254,674 raw frames), V2 removed all selections that fail its candidate-quality rule, reduced the continuity proxy by 76.92%, reduced HIGH-risk clips from 253 to 76, and introduced a 0.2187% explicit-missing rate. The audit passed before training was started. In the subsequent one-seed controlled comparison, validation accuracy increased by 1.77 percentage points and macro-F1 by 2.19 points.

This is evidence of better skeleton quality and temporal continuity, not evidence that the selected track is always the child. MMASD OpenPose still has no verified persistent identity or child/therapist role labels.

## A. Tracker V1 recap

The original implementation in `src/data/child_tracking.py` is unchanged. Legacy mode still uses the existing frame-initialized score and greedy temporal selection, including its pre-existing smaller-person preference. The new dispatcher defaults to `legacy` when `tracking` is absent, so old configs remain valid (`src/data/tracking.py:13-34`).

A bitwise sanity check compares direct legacy output with dispatched legacy output on three real manifest clips, including selected indices, failures, and every selected skeleton array (`tests/test_tracklet_tracking.py:106-125`).

## B. Tracker V2 design

V2 separates two stages:

1. Build anonymous, quality-filtered person tracklets across the entire clip.
2. Rank completed tracklets using neutral track-level quality and continuity signals, then emit exactly one target track.

Each track stores its last and previous skeleton, center, scale, confidence, valid mask, observations, age through frame indices, lost-frame count, lost events, and termination state (`src/data/tracklet_tracking.py:58-126`). It uses velocity-extrapolated center prediction when two observations exist. Assignment is deterministic greedy over all gated edges sorted by cost, with strict one-track/one-candidate constraints (`src/data/tracklet_tracking.py:322-374`). No SciPy dependency was added.

## C. Candidate filtering

A candidate must have:

- at least 10 valid BODY_25 joints;
- at least 4 of Neck, MidHip, L/R Shoulder, and L/R Hip;
- finite coordinates/confidences and finite derived geometry;
- bounding-box diagonal at least 50 px.

Implementation: `candidate_from_person` at `src/data/tracklet_tracking.py:162-205`. These thresholds are configurable and recorded in `configs/paper_repro_tracker_v2.yaml:8-11`.

The geometry-only calibration used 100 clips (10 manifest-order-spaced clips per class), 17,870 frames, and no activity labels in scoring or validation accuracy. Candidate quantiles were: valid joints p10=9, torso anchors p10=4, and diagonal p05=87.08 px. The chosen rule retained 57,771/65,143 legacy-valid candidates (88.68%) while leaving zero sampled frames without any V2 candidate. Full calibration is in `tracker_v2_audit/candidate_filter_calibration.json` and is reproducible with `scripts/calibrate_tracklet_filter.py`.

The 50 px rule is a fragment floor, not a comparison that prefers one person over another. A genuinely distant/small but coherent skeleton could still be filtered; this remains a documented limitation.

## D. Matching cost and gating

For each track-candidate edge:

`cost = 0.40*d_center + 0.35*d_pose + 0.20*d_scale + 0.05*(1-confidence)`

where center displacement is relative to predicted center and normalized by body diagonal; pose distance uses shared joints centered on torso and normalized by each skeleton's diagonal; and scale change is the absolute log diagonal ratio (`src/data/tracklet_tracking.py:208-244`).

Hard gates reject an edge when any condition fails:

- normalized center distance > 0.50 (relaxed by sqrt(frame gap));
- scale ratio > 1.80;
- normalized pose distance > 0.35 (relaxed by sqrt(frame gap));
- shared valid joints < 6.

Weights and gates are in `configs/paper_repro_tracker_v2.yaml:13-24`. Scale is used for normalization and temporal consistency, not as “smaller = child.”

## E. Track initialization / lost / reacquisition

Every unmatched valid candidate initializes an anonymous track; the first frame does not choose a child. Unmatched active tracks become lost. With `max_lost_frames: 3`, a track can be reacquired after one to three missing frames if all gates pass; it terminates after the allowance is exceeded (`src/data/tracklet_tracking.py:105-126,322-374`). An impossible jump creates a new track instead of forcing a match.

Once the target track is chosen, absent observations are emitted as index `-1` and person `None`; V2 does not substitute another person (`src/data/tracklet_tracking.py:438-484`). Existing preprocessing receives that explicit missing sequence afterward and performs its unchanged interpolation/normalization (`src/data/dataset.py:129-142`).

## F. Target-track selection

Completed tracks are scored using:

| Component | Weight |
|---|---:|
| coverage | 0.45 |
| mean valid-joint fraction | 0.20 |
| mean confidence | 0.15 |
| continuity | 0.10 |
| scale stability | 0.05 |
| low fragmentation | 0.05 |

The score is implemented at `src/data/tracklet_tracking.py:377-421`. The optional bounded smaller-size prior has weight 0.02 but is **disabled** for this experiment (`use_size_prior: false`). Therefore target selection has no active smaller-person prior. The output should be described as a selected target track, not a child classification.

## G. Config added

- `configs/paper_repro_tracker_v1.yaml`: `tracking.mode: legacy`.
- `configs/paper_repro_tracker_v2.yaml`: complete filter, matching, gating, lost-track, and target-selection settings.

The common dispatcher is wired at `src/data/tracking.py:13-34`, `src/data/dataset.py:129`, and `src/engine.py:27-40`. Checkpoint verification now treats tracking configuration as an inference-critical input (`src/reproducibility.py:24-37`), with a regression test at `tests/test_reproducibility.py:42-48`.

Resolved V1/V2 configs are identical outside `tracking` and experiment-specific checkpoint/run output directories. They share the same artifacts and split directories. This is asserted by `scripts/summarize_tracker_training.py:53-119` and recorded in `tracker_training/comparison_summary.json`.

## H. Tests

All **40 repository tests pass** (`pytest -q`). The new sanity suite covers:

- A: fragment versus full skeleton;
- B: smooth single-person motion;
- C: two people crossing with candidate order changes;
- D: raw OpenPose candidate-index reordering;
- E: loss for one to three frames and gated reacquisition without substitution;
- F: impossible jump is not force-matched;
- G: one-person clip;
- real-data bitwise legacy compatibility on three clips;
- rejection of evaluation with a mismatched tracker mode.

Cases A–G and legacy compatibility are at `tests/test_tracklet_tracking.py:44-125`.

## I. V1 vs V2 audit metrics

Audit command: `python scripts/compare_trackers.py`

The same frozen manifest and the same earlier proxy thresholds were used: normalized centroid threshold 0.1636905059 and absolute log-scale threshold 0.3438655165. A fragment is a selected skeleton that fails the declared V2 quality rule. HIGH risk means a clip has a continuity-proxy event or at least 5% missing target frames. Definitions and PASS gates are frozen in `scripts/compare_trackers.py:147-232`.

| Metric | V1 | V2 | Delta V2−V1 | Relative reduction |
|---|---:|---:|---:|---:|
| fragment selection | 4,551 frames (1.7870%) | 0 (0.0000%) | −4,551; −1.7870 pp | 100.00% |
| HIGH-risk clips | 253 | 76 | −177 | 69.96% |
| centroid jumps | 1,489 (0.5878%) | 381 (0.1508%) | −1,108; −0.4371 pp | 74.35% |
| scale jumps | 1,521 (0.6005%) | 378 (0.1496%) | −1,143; −0.4509 pp | 75.09% |
| continuity proxy | 1,390 (0.5488%) | 320 (0.1266%) | −1,070; −0.4221 pp | 76.92% |
| clips with continuity proxy | 253 | 71 | −182 | 71.94% |
| missing target | 0 (0.0000%) | 557 (0.2187%) | +557; +0.2187 pp | — |

All predefined gates passed: ≥80% fragment reduction, ≥30% reduction in HIGH risk/centroid/scale, and ≤5% missing rate. Detailed outputs are `tracker_v2_audit/summary.json`, `aggregate_comparison.csv`, `per_clip_v1_v2.csv`, and `per_activity_v1_v2.csv`.

## J. Visual inspection findings

Twenty paired skeleton-canvas comparisons were generated around the exact anomaly frames from the previous top-20 suspicious list. The complete index is `tracker_v2_audit/visual_comparison_manifest.csv`; images are under `tracker_v2_audit/visual_comparisons/`.

Six representative pairs were manually spot-checked across MFS, drumming, chest expansion, and squat. They show V2 following a coherent full skeleton while V1 changes raw candidate index or selects a fragment. In the required clip `mfs_40504_D16_001_n`, V1 selects the tiny fragment around frame 23 and then jumps to a full skeleton; V2 selects a full skeleton consistently through frames 21–25. See `tracker_v2_audit/visual_comparisons/mfs_40504_D16_001_n_around_23.png`.

Candidate index differences alone are not interpreted as identity switches, and skeleton-only canvases cannot establish child versus therapist role.

## K. Controlled training comparison

Training was started only after the audit status was PASS. Both runs used seed 42, the same 1,091 training clips and 283 validation clips, fixed full-clip-to-60 sampling, preprocessing, model (860,362 trainable parameters), optimizer/loss/hyperparameters, manifest, and split. Execution was CPU-only (Python 3.13.11, PyTorch 2.5.1+cu121, CUDA unavailable).

Exact commands:

```bash
python -m src.train --config configs/paper_repro_tracker_v1.yaml
python -m src.evaluate --config configs/paper_repro_tracker_v1.yaml --checkpoint checkpoints/tracker_v1_controlled/best.pt --split val --output-dir artifacts/tracker_training/v1
python -m src.train --config configs/paper_repro_tracker_v2.yaml
python -m src.evaluate --config configs/paper_repro_tracker_v2.yaml --checkpoint checkpoints/tracker_v2_controlled/best.pt --split val --output-dir artifacts/tracker_training/v2
```

| Metric | V1 legacy | V2 tracklet | Delta V2−V1 |
|---|---:|---:|---:|
| best epoch | 53 | 11 | −42 |
| train accuracy at best epoch | 95.3254% | 87.4427% | −7.8827 pp |
| train macro-F1 at best epoch | 93.8915% | 84.1000% | −9.7916 pp |
| validation accuracy | 69.2580% | 71.0247% | **+1.7668 pp** |
| macro precision | 66.8649% | 72.2959% | +5.4310 pp |
| macro recall | 74.2753% | 72.8314% | −1.4440 pp |
| macro-F1 | 67.7455% | 69.9385% | **+2.1930 pp** |
| validation loss | 1.9971 | 1.1703 | −0.8268 |

Per-class validation recall/F1:

| Activity (support) | Recall V1 | Recall V2 | Δ recall | F1 V1 | F1 V2 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| arm_swing_as (33) | 81.82% | 78.79% | −3.03 pp | 71.05% | 67.53% | −3.52 pp |
| body_swing_bs (30) | 53.33% | 83.33% | +30.00 pp | 61.54% | 84.75% | +23.21 pp |
| chest_expansion_ce (30) | 56.67% | 83.33% | +26.67 pp | 68.00% | 80.65% | +12.65 pp |
| drumming_dr (62) | 69.35% | 66.13% | −3.23 pp | 72.88% | 73.21% | +0.33 pp |
| maracas_forward_shaking_mfs (26) | 61.54% | 34.62% | −26.92 pp | 61.54% | 50.00% | −11.54 pp |
| maracas_shaking_ms (40) | 75.00% | 70.00% | −5.00 pp | 80.00% | 73.68% | −6.32 pp |
| sing_and_clap_sac (35) | 68.57% | 71.43% | +2.86 pp | 63.16% | 62.50% | −0.66 pp |
| squat_sq (17) | 76.47% | 82.35% | +5.88 pp | 74.29% | 70.00% | −4.29 pp |
| tree_pose_tr (4) | 100.00% | 75.00% | −25.00 pp | 50.00% | 46.15% | −3.85 pp |
| twist_pose_tw (6) | 100.00% | 83.33% | −16.67 pp | 75.00% | 90.91% | +15.91 pp |

Machine-readable per-class deltas are in `tracker_training/per_class_comparison.csv`. Confusion matrices are:

- V1: `tracker_training/v1/confusion_matrix.png`, with count/percent CSVs beside it.
- V2: `tracker_training/v2/confusion_matrix.png`, with count/percent CSVs beside it.
- V2−V1 cell-count delta: `tracker_training/confusion_matrix_delta_counts.csv`.

Interpretation must remain cautious: this is one fixed seed, no significance estimate, and several classes have very small validation support. V2 notably improves body swing and chest expansion but reduces MFS recall. Thresholds were not selected using these results.

Checkpoint SHA-256:

- V1: `531374993417e636695f8f53592813cbbd8bddb3e35bb89a8fa59fdfb7088b86`
- V2: `3f99d7a6f312e6605814a9d616d33bc6c5a7a25d3ca90e88554792f8b5d176bc`

Manifest/split SHA-256 shared by both runs:

- manifest: `d317ddcd17633fba1f5186223afbc7ec8a91bbf0f894d5c9d1edf1dd11f8f65d`
- train split: `81fdb486982839662e72dc85f4cf1c665b77cb4eb3c4adb3b5a514ecfa0449ad`
- validation split: `2b21991c7d0db6a076a395d664ea2b0cf8928ed517b3a941b9022c578fd2f888`

After training/evaluation, `src/reproducibility.py` received one safety-only change: tracker mode/config was added to checkpoint evaluation compatibility checks. The checkpoints retain their exact training-time source hashes and complete resolved tracker configs. No model input values or reported metrics were recomputed by that guard change; a future evaluation against the current source will correctly report the post-training source-file difference.

## L. Final recommendation

Freeze this V2 configuration as the tracking candidate for the next audit and proceed to **Công việc 3 — 60-frame sequence construction audit**. Keep V1 as the baseline and retain both experiment artifacts. Do not describe V2 as verified child tracking until persistent identity/role ground truth or a validated manual/RGB review exists.

The key follow-up during Work 3 is to inspect how 557 explicit missing detections are interpolated by the unchanged preprocessing and how uniform full-clip-to-60 sampling represents short/high-motion segments. The MFS class regression deserves descriptive error analysis, but it must not be used to tune tracker thresholds against validation accuracy.

## Seven direct answers

1. **Tiny skeleton fragments đã được loại hiệu quả chưa?** Có theo operational definition: 4,551 selected fragment frames xuống 0, giảm 100%. This is proxy-defined, not identity ground truth.
2. **Temporal discontinuity giảm bao nhiêu?** Continuity-proxy rate giảm 76.92%; centroid-jump rate giảm 74.35%; scale-jump rate giảm 75.09%.
3. **HIGH-risk clips giảm từ bao nhiêu xuống bao nhiêu?** 253 xuống 76, giảm 177 clip (69.96%).
4. **Missing-frame rate có tăng đáng kể không?** Tăng từ 0% lên 0.2187% (557/254,674 frames); this is small and well below the predefined 5% gate, while preserving honest missingness instead of switching person.
5. **V2 có còn strong size bias không?** Không trong target selection: `use_size_prior=false`. Size remains only as a fragment floor, normalization scale, and scale-continuity signal.
6. **Validation accuracy/F1 thay đổi thế nào?** Accuracy tăng 69.2580% → 71.0247% (+1.7668 pp); macro-F1 tăng 67.7455% → 69.9385% (+2.1930 pp), for one fixed seed.
7. **Đủ ổn để chuyển sang Công việc 3 chưa?** Có, với status PASS and the caveat that selected-person role is still unverified.

## Files changed/created for this work

Implementation/config/tests/scripts:

- `src/data/tracklet_tracking.py` (new)
- `src/data/tracking.py` (new)
- `src/data/dataset.py` (modified dispatcher call only)
- `src/engine.py` (modified config injection)
- `src/reproducibility.py` (modified inference provenance guard)
- `configs/paper_repro_tracker_v1.yaml` (new)
- `configs/paper_repro_tracker_v2.yaml` (new)
- `tests/test_tracklet_tracking.py` (new)
- `tests/test_reproducibility.py` (modified)
- `scripts/compare_trackers.py` (new)
- `scripts/calibrate_tracklet_filter.py` (new)
- `scripts/summarize_tracker_training.py` (new)

Generated audit artifacts:

- `artifacts/tracker_v2_audit/{summary.json,aggregate_comparison.csv,per_activity_v1_v2.csv,per_clip_v1_v2.csv,candidate_filter_calibration.json,visual_comparison_manifest.csv}`
- 20 PNG files listed exactly by `artifacts/tracker_v2_audit/visual_comparison_manifest.csv` under `visual_comparisons/`

Generated training artifacts:

- `artifacts/tracker_training/{comparison_summary.json,per_class_comparison.csv,confusion_matrix_delta_counts.csv}`
- `artifacts/tracker_training/v1/{validation_results.json,predictions.csv,per_class_metrics.csv,confusion_matrix_counts.csv,confusion_matrix_percent.csv,confusion_matrix.png}`
- `artifacts/tracker_training/v2/{validation_results.json,predictions.csv,per_class_metrics.csv,confusion_matrix_counts.csv,confusion_matrix_percent.csv,confusion_matrix.png}`
- `checkpoints/tracker_v1_controlled/best.pt`
- `checkpoints/tracker_v2_controlled/best.pt`
- `runs/tracker_v1_controlled/history.json`
- `runs/tracker_v2_controlled/history.json`
- `artifacts/TRACKER_V2_FINAL_REPORT.md` (this report)

Raw skeleton data, old tracker implementation, old artifacts, manifest, splits, sequence construction, normalization, model architecture, optimizer, loss, and training hyperparameters were not modified.
