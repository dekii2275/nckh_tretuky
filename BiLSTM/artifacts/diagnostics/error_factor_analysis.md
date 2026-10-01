# Descriptive S1 validation error factors

Only the 275 retained validation clips contribute to correct/incorrect comparisons. Eight short clips remain in the per-clip CSV with blank prediction fields. Rates are fractions 0–1; confidence is the maximum softmax of mean logits, not calibrated probability.

Keypoint confidence is averaged over all 25 selected raw joints per frame, including zero-confidence/missing joints and zero for no selected candidate. Mean valid-joint confidence is also reported. num_people is the time mean of raw OpenPose candidate counts, not distinct verified identities.

| classification | clips | mean_missing_joint_rate | mean_torso_fallback_rate | mean_mean_keypoint_confidence | mean_num_people | mean_prediction_confidence | human_reviewed_clips |
|---|---|---|---|---|---|---|---|
| correct | 204 | 0.083 | 0.005 | 0.663 | 3.442 | 0.966 | 0 |
| incorrect | 71 | 0.206 | 0.021 | 0.547 | 3.680 | 0.873 | 0 |

These are unadjusted clip-weighted descriptions, with per-activity breakdown in the CSV. Differences can reflect class composition, duration, detection noise and other confounding. A changing raw selected index is not a verified identity switch. Without human labels, tracking ambiguity/error cannot be measured from it.
