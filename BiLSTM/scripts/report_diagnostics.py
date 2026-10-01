#!/usr/bin/env python3
"""Assemble the diagnostic decision report without training or selecting protocols."""
import json

from diagnostics_common import OUT, REVIEW, ROOT, json_dump, markdown_table, preserve_runs, read_labels, read_manifest, write_csv


def main():
    protected = preserve_runs()
    gaps = read_manifest(OUT / "per_class_gap_ranked.csv")
    for row in gaps:
        for k in ("paper_recall", "s1_clip_recall", "s3_clip_recall", "paper_minus_s1_clip", "paper_minus_s3_clip"):
            row[k] = float(row[k])
    quality = json.loads((OUT / "preprocessing_usage_summary.json").read_text())
    tracking = json.loads((OUT / "tracking_audit_summary.json").read_text())
    size = json.loads((OUT / "child_size_assumption_summary.json").read_text())
    assets = json.loads((OUT / "external_tracking_assets.json").read_text())
    factors = read_manifest(OUT / "error_factor_comparisons.csv")
    overall_factors = [r for r in factors if r["activity"] == "ALL"]
    rates = tracking["overall"]
    rate_note = "All A/B/C/D rates are unavailable, not zero error rates." if not rates["reviewed"] else "Rates among reviewed clips: " + ", ".join(f"{k}={rates[k]:.2%}" for k in ("correct_child_rate", "wrong_person_rate", "identity_switch_rate", "uncertain_rate")) + "."
    size_note = "Percent child-smaller/larger/ambiguous is unavailable." if not size["reviewed_clips_with_pairs"] else "Clip-weighted size results: " + json.dumps(size["results"]) + "."
    primary = "1. improve child selection"
    rationale = "Complete the prepared human child/person identity audit first; only then decide whether the tracker needs correction. Zero human labels cannot establish that tracking is good or bad. Do not modify the tracker yet."
    if tracking["split_sensitivity_gate_passed"]:
        correct = next(r for r in overall_factors if r["classification"] == "correct")
        incorrect = next(r for r in overall_factors if r["classification"] == "incorrect")
        if float(incorrect["mean_missing_joint_rate"]) - float(correct["mean_missing_joint_rate"]) >= 0.05:
            primary = "3. audit missing-joint preprocessing"
            rationale = "The human tracking gate is satisfied and the unadjusted missingness difference is at least 5 percentage points. Investigate missing-joint handling without automatically changing it; this association is not causal proof."
        else:
            primary = "2. test split sensitivity"
            rationale = "The human tracking gate is satisfied; prepare the fixed seeds 0,1,2,3,4 using S1 and report all results. No seed selection by validation accuracy. This report does not launch training."
    lines = ["# Reproduction diagnostics after R0/S1/S2/S3", "",
             "The published architecture remains frozen at 860362 trainable parameters. No source model, preprocessing implementation, training configuration or frozen split was changed, and no model was trained in this phase. R0/S1/S2/S3 artifacts are protected by SHA256.", "",
             "## A. Per-class recall gap", "",
             "Paper diagonal recalls below were supplied by the user. Recall is %, gap is percentage points; the table is ranked by the larger S1/S3 gap. R0 and window-level recalls/supports are also in per_class_gap.csv.", ""]
    lines += markdown_table(gaps, ["class_name", "paper_recall", "s1_clip_recall", "s3_clip_recall", "paper_minus_s1_clip", "paper_minus_s3_clip", "support_s1_clip"])
    lines += ["", "The largest raw recall deficits are Maracas Forward Shaking, Chest Expansion, Arm Swing, Body Swing and Drumming. Drumming contributes the most support-weighted deficit in both S1 and S3 (4.40 and 5.49 points against paper recalls applied to our validation distribution). This is not a decomposition of the published 94.84% accuracy, whose class supports are unknown here.",
              "", "Maracas Shaking and Sing and Clap are within about 2.2 recall points of the supplied diagonal in both protocols; S3 Squat is within 1.90 points. No cause follows from per-class recall alone.",
              "", "## B. Confusion structure", ""]
    confusions = read_manifest(OUT / "confusion_comparison.csv")
    for r in confusions:
        for k in list(r):
            if "percent" in k:
                r[k] = float(r[k])
    lines += markdown_table(confusions, ["class_true", "class_pred", "paper_percent", "r0_percent", "s1_clip_percent", "s3_clip_percent"])
    top = read_manifest(OUT / "top20_confusion_pairs.csv")
    for key in ("s1_clip", "s3_clip"):
        selected = [r for r in top if r["experiment_level"] == key][:5]
        for r in selected:
            r["percent"] = float(r["percent"])
        lines += ["", f"Top five {key} confusions by number of errors:", ""]
        lines += markdown_table(selected, ["class_true", "class_pred", "count", "row_support", "percent", "paper_percent"])
    lines += ["", "All top-20 nonzero pairs at window and clip levels are in top20_confusion_pairs.csv. Only three off-diagonal paper entries are supplied; structural similarity of the full paper matrix is therefore not established. Similar selected entries do not prove reproduction.",
              "", "## C. Human tracking audit", "",
              f"Reviewed {rates['reviewed']}/120. A={rates['A_count']}, B={rates['B_count']}, C={rates['C_count']}, D={rates['D_count']}. {rate_note} Current status: {tracking['status']}.",
              "", "The existing 120-clip raw-frame set is reused. index.html and each clip page support A/B/C/D, secondary flags, CSV export/import, candidate indices, selected index, bounding boxes, MidHip and confidence. Previous browser label storage is not converted into human A/B/C/D labels automatically.",
              "", "tracking_vs_accuracy.csv will compare saved S1 validation accuracy/F1 for A/B/C/D and B+C once labels exist. It excludes training clips and short clips without S1 validation predictions. Rates by activity/session/subject/max-person-count are in tracking_label_breakdown.csv. The deliberately stratified review set is not an unbiased whole-dataset prevalence sample.",
              "", "## D. Child-size assumption", "",
              f"Status: {size['status']}; {size['confident_measured_pairs']} confidently human-identified frame pairs. {size_note} manual_person_pairs.csv and the UI let a reviewer identify child/therapist raw indices at the same frame. analyze_child_size.py then measures original bbox area and skeleton height, with a predeclared +/-5% ambiguous band and clip-weighted median ratios.",
              "", "A/B/C/D alone does not identify both people's boxes. No role is inferred from size, appearance or the external ASD_Label. The existing smaller-person prior remains unchanged. No oracle subset test or candidate tracker is prepared/run without the required human evidence.",
              "", "## E. Local tracking assets", "",
              f"Read-only inventory found {assets['mediapipe_csv_files']} MediaPipe CSVs, {assets['optical_flow_mp4_files']} optical-flow MP4s and {assets['romp_npz_frames']} ROMP NPZ frames. Stored ASD_Label file counts: {assets['ASD_Label_file_counts']}. Values beyond 0/1 have unclear semantics in the inspected local code.",
              "", f"Filename candidates cover {assets['frozen_clips_with_filename_match']}/1374 frozen clips, but all MediaPipe CSVs have 179 data rows and none of the matched CSVs has an equal raw OpenPose frame count. Crop-to-original coordinates and frame/person mappings are unverified. README/code describe YOLO/DeepSORT and person labels; feature outputs and existing labels are not a validated child-ID mapping. See external_tracking_assets.md for evidence and limitations.",
              "", "Validated external role metadata could eventually select original 2D OpenPose persons while preserving 60×50 input. Feeding MediaPipe 3D, ROMP or optical-flow features would change the input protocol and is not proposed.",
              "", "## F. Actual preprocessing usage", "",
              "All 1374 frozen clips are audited before temporal sampling. Frame/joint rates are pooled by split, not means of clip percentages. The implementation is unchanged.", ""]
    qrows = []
    for split, row in quality["by_split"].items():
        qrows.append({"split": split, "frames": row["frames"], "missing_MidHip_%": row["invalid_midhip_percent"],
                      "missing_Lshoulder_%": row["missing_left_shoulder_percent"], "missing_Rshoulder_%": row["missing_right_shoulder_percent"],
                      "invalid_scale/torso_fallback_%": row["torso_fallback_percent"], "filled_missing_joints_%": row["interpolated_joints_percent"],
                      "bounded_interpolation_%": row["bounded_interpolated_joints_percent"], "edge_fill_%": row["edge_filled_joints_percent"],
                      "frames_gt25%missing_%": row["frames_gt25pct_joints_missing_percent"], "severe_clips_%": row["severe_missingness_clips_percent"]})
    lines += markdown_table(qrows, list(qrows[0]))
    lines += ["", "Severe missingness is a predeclared diagnostic flag: more than 25% of clip frames have more than 25% invalid joints. Filled joints includes bounded interpolation and leading/trailing extension, but excludes never-observed joints. Torso fallback includes its existing span/1.0 secondary safeguards. Per-class tables, original scale counters and full run-length distributions are saved separately.",
              "", "## G. Validation error factors", "",
              "S1 mean-logit clip predictions: 204 correct / 71 incorrect on 275 retained validation clips. The error table preserves all 283 validation clips, with prediction fields blank for the eight dropped short clips.", ""]
    erows = []
    for r in overall_factors:
        erows.append({"classification": r["classification"], "clips": r["clips"],
                      "missing_joints_%": 100 * float(r["mean_missing_joint_rate"]), "torso_fallback_%": 100 * float(r["mean_torso_fallback_rate"]),
                      "mean_keypoint_confidence": float(r["mean_mean_keypoint_confidence"]), "mean_people": float(r["mean_num_people"]),
                      "prediction_confidence": float(r["mean_prediction_confidence"]), "logit_margin": float(r["mean_prediction_logit_margin"])})
    lines += markdown_table(erows, list(erows[0]))
    lines += ["", "Errors are descriptively associated with higher missingness/fallback frequency, lower keypoint confidence and a slightly larger mean raw candidate count. Prediction confidence is uncalibrated softmax of mean logits. Tracking ambiguity is unmeasured until human labels exist; raw-index changes cannot substitute for identity-switch labels. Class composition and other confounders remain.",
              "", "## Current split limitation", "",
              "Tree Pose: 4 validation clips (3.10% of that class); Twist Pose: 6 (5.00%). Both use a single validation subject/session, 41023/D8. These small supports make recalls unstable. The current split was not modified. No future seed candidates were generated because the human tracking-quality gate has not passed. Future seeds remain predeclared as 0,1,2,3,4 with S1, never selected by accuracy.",
              "", "## H. Exactly one primary next step", "", primary, "", rationale,
              "", "The automatic diagnostics and review tooling are complete. Human identity review and human role-pair measurements remain pending, so the primary cause of the remaining paper gap is not yet determined.",
              "", "## Files and rerun commands", "",
              "Open `../tracking_review/index.html`. Export labels/pairs, replace `../tracking_review/manual_labels.csv` and `manual_person_pairs.csv`, then run from BiLSTM:", "", "```bash",
              "python scripts/analyze_tracking_labels.py", "python scripts/analyze_child_size.py", "python scripts/report_diagnostics.py", "```", "",
              f"Preservation check: {protected} protected experiment/config/source files unchanged. No full retraining, architecture tuning, tracker replacement, oracle test or seed sweep occurred."]
    (OUT / "diagnostic_report.md").write_text("\n".join(lines) + "\n")
    json_dump({"automatic_diagnostics": "complete", "human_review": tracking["status"], "child_size": size["status"],
               "primary_next_step": primary, "rationale": rationale, "protected_files_unchanged": protected,
               "training_performed": False, "model_parameters": 860362, "tracker_modified": False,
               "split_candidates_generated": False, "remaining_gap_cause_established": False}, OUT / "diagnostic_status.json")
    print(primary, "—", rationale)


if __name__ == "__main__":
    main()
