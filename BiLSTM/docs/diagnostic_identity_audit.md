# Diagnostic phase after R0/S1/S2/S3

The completed automatic report is
[`artifacts/diagnostics/diagnostic_report.md`](../artifacts/diagnostics/diagnostic_report.md).
It separates measured class/split/preprocessing findings from missing human
identity evidence. No model training, source preprocessing change, tracker change,
new split, or hyperparameter tuning occurs in this phase.

From `BiLSTM`, using the existing `mmasd-bilstm` Python environment:

```bash
python scripts/analyze_per_class_gap.py
python scripts/prepare_manual_review.py
python scripts/analyze_tracking_labels.py
python scripts/analyze_child_size.py
python scripts/audit_external_tracking_assets.py
python scripts/audit_preprocessing_usage.py
python scripts/report_diagnostics.py
```

The preprocessing audit rereads all raw clips using the unchanged production
dataset/preprocessing path. External assets are inspected in place, without
downloads, archive extraction or data substitution. Protected experiment/config/
source hashes are checked before and after scripts.

## Human review

Open `artifacts/tracking_review/index.html` in a browser. Each clip shows all raw
frames and detected skeletons, the tracker selection, raw index, MidHip, bounding
box, and keypoint confidence. Review the whole clip, including index-reordering
events. These are skeleton-only views; seek RGB/context when role identification
is ambiguous, and label D rather than guessing from apparent size.

- A: clearly tracks the child correctly.
- B: clearly tracks a therapist or another wrong person.
- C: starts correctly but switches identity.
- D: uncertain or impossible to determine.
- Blank: not yet reviewed; it is not D and not a tracking success.

Secondary flags are optional: blank = not assessed, 1 = yes, 0 = no. The CSV
contains all required columns plus child-leaves-frame, therapist-crosses-child,
severe-keypoint-failure, tiny-skeleton and large-camera-motion flags.

The UI saves browser drafts when local storage is available. Export all labels
from the index, or export individual clips and import their CSVs into the index
to merge them. Replace `artifacts/tracking_review/manual_labels.csv` with the
exported file. Browser edits alone do **not** update the filesystem CSV. Existing
legacy browser labels are preserved in their old namespace and not automatically
translated into A/B/C/D. CSV revision namespaces prevent stale browser drafts
from silently superseding a newly imported filesystem label version.

The optional frame-pair panel records the raw index of a manually recognized
child and therapist at the same frame. Check role certainty and both-visible
only when supported by human review. Export `manual_person_pairs.csv`. Indices
are zero based and frame-local; reordering means they are not persistent IDs.
This file supports bbox-area/height measurements, not a complete oracle track.

After replacing downloaded files:

```bash
python scripts/analyze_tracking_labels.py
python scripts/analyze_child_size.py
python scripts/prepare_manual_review.py
python scripts/report_diagnostics.py
```

Tracking rates use reviewed clips as denominator, including D. S1 performance
by A/B/C/D uses only held-out retained clips with saved predictions and macro-F1
over the fixed 10 classes; subgroup class support is reported. The 120-clip
stratified, high-reordering sample is not a prevalence estimate for all clips.
The persons category in clip-level breakdown is the maximum raw candidate count.

Size measurements use human-confirmed roles only, confidence >=0.1 valid original
OpenPose joints, bbox area and vertical extent. The primary unit is a clip's
median confirmed frame-pair ratio. A +/-5% ratio band and unmeasurable/unconfirmed
cases are ambiguous. Without human annotations, outputs are blank/null, not
assumed child-smaller rates.

## Conditional work deliberately deferred

Candidate tracker corrections require meaningful human wrong/switch evidence.
An oracle subset needs confident child indices for the necessary frames and
separate labeling; none is run from clip-level labels alone.

Only after a completed human review with correct-child >=95%, wrong-person <=3%
and switches <=3% should split sensitivity be prepared. The future seed list is
exactly 0,1,2,3,4 using S1 stride60. No candidate splits/configs or training have
been generated now. The conservative complete-review requirement avoids treating
one or two favorable reviews as evidence of a high-quality tracker.

Missingness/error comparisons are descriptive and unadjusted, with per-class
tables available. They cannot separate tracking error from occlusion, detection
quality, class composition, or other confounding. The single primary next step
is human child/person verification before deciding on a selection correction.
