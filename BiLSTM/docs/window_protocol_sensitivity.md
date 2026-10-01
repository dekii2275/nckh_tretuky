# Frozen sequence-construction sensitivity study

This extends the existing reproduction without changing the paper model. R0 remains
in `artifacts/paper_repro`, `runs/paper_repro`, and `checkpoints/paper_repro`.
`artifacts/window_audit/protected_hashes.json` records its files and the unchanged
model, parser, tracking, preprocessing, seed, configuration and split files.

Run from `BiLSTM` with the existing `mmasd-bilstm` environment:

```bash
python scripts/run_window_study.py          # counts, guards, all three smoke tests
python scripts/run_window_study.py --train  # same prerequisites, then S1/S2/S3 on CUDA
python scripts/audit_confusion_granularity.py
python scripts/report_window_study.py
python scripts/verify_window_study.py       # verify all completed saved artifacts
python scripts/export_tracking_review.py   # manual identity review preparation
```

CUDA is required for full training. No automatic environment installation or CPU
fallback is performed. A sandbox that hides NVIDIA devices must allow the process
to access the existing GPU. Each experiment refuses to overwrite an existing
checkpoint or partial training history. A complete history may be reused by the
study runner; evaluation verifies its checkpoint configuration and input hashes.

The existing clip split is loaded and checked before window expansion. Each
window is `full_preprocessed_clip[start:start+60]`, with starts `0, stride, ...`
while the end fits. Strides 60, 30 and 15 are predeclared. Short clips and tails are
dropped. Tracking and missing-joint interpolation still run on each full clip,
preserving R0 preprocessing; there is no temporal resampling inside windows.
Window frame offsets are zero based; metadata `end_frame` is inclusive.
Source JSON count mismatches cause an error when a clip is loaded.

Model parameter counts are BiLSTM 579584, MHA 263168, LayerNorm 512, projection
16448 and classifier 650, totaling 860362. The same seed, Adam settings, batch 32,
CrossEntropyLoss, 100-epoch ceiling and accuracy-based patience 10 are retained.
Longer clips contribute more training windows, and overlapping windows are
correlated; both are part of this construction sensitivity.

Checkpoint selection uses validation **window accuracy**, following the existing
sample-level accuracy monitor. At that checkpoint, evaluation saves every window's
metadata and logits, and averages logits over all windows of each retained clip.
Both levels save accuracy, macro precision/recall/F1, integer confusion matrices,
row-normalized percentages and per-class support. No alternative aggregation or
stride is chosen after observing scores.

Epoch loss is the sample-weighted mean of batch CrossEntropyLoss; the additional
batch-loss sum adds the unweighted batch means, with batch count reported. The
optimization reduction is unchanged. R0's historical sum is unavailable because
it was not logged; multiplying its mean by its batch count would not recover it
exactly due to its partial final batch.

The main deliverables are `artifacts/protocol_sensitivity.csv` and
`artifacts/protocol_sensitivity_report.md`. Detailed counts, dropped tails,
per-class and per-session-group totals, window metadata, CUDA/smoke reports,
granularity and paper-pattern diagnostics are in `artifacts/window_audit`.
Since eight short validation clips are dropped, R0 predictions on the same 275
retained clips are also reported as a coverage control, without retraining R0.

This is a protocol sensitivity study, not evidence of the author's true stride.
The supplied paper accuracy and confusion percentages are comparison references.
Remaining unspecified choices and historical provenance limitations are listed
in the generated report. If windows fail to explain the gap, the next diagnosis
is manual tracked-person identity review, with no automatic tracker redesign or
architecture tuning.

The completed S1/S2/S3 study and 120-clip manual review set are linked from
`artifacts/protocol_sensitivity_report.md`. Review pages are standalone HTML with
all raw skeleton frames and no RGB background. The full raw population has no
clip containing only one candidate throughout; five clips contain some one-person
frames, of which three are in the selected set. Reordering is a geometric proxy
for review sampling, not proof of identity switches.
