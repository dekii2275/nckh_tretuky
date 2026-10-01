# MMASD BiLSTM + Attention Baseline

Reimplementation baseline for **“A BiLSTM with Attention Approach Using 2D
Skeletons for Autism Activity Recognition”** using the local MMASD OpenPose
modality.

## Location

This baseline is self-contained in `BiLSTM/`. The local dataset remains at the
repository level and is referenced as `../data_MMASD`; it is not copied into
this directory.

Run the commands below from either the repository root or `BiLSTM/`. The
scripts resolve paths from the baseline configuration, so both locations are
supported.

## Objective

The pipeline is deliberately small and reproducible:

```text
MMASD OpenPose BODY_25
  -> deterministic multi-person tracking candidate
  -> MidHip centering
  -> shoulder-width normalization
  -> BiLSTM
  -> masked multi-head self-attention
  -> masked temporal mean pooling
  -> activity classifier
```

Only `../data_MMASD/2D_skeleton/2D_openpose` is used. RGB, optical flow, 3D
skeletons, and ADOS scores are not model inputs.

## Dataset inspection findings

The raw MMASD dataset is expected to already exist locally and is intentionally
ignored by git. It is never downloaded, copied, moved, or modified by this
project.

The checked-in inspection scripts found:

| Property | Observed locally |
|---|---:|
| Activity directories | 11 |
| Clip directories | 1,693 |
| Non-empty OpenPose clips | 1,374 |
| Subjects | 32 |
| Session IDs | `D1`, `D8`, `D16` |
| Subject-session groups | 75 |
| OpenPose pose format | BODY_25, 25 joints, `[x,y,confidence]` |
| People per frame | 1–9 |

The repository README reports 1,315 samples and 168 Drumming clips, but this
local OpenPose tree contains 545 Drumming directories, of which 343 are
non-empty. Frog Pose has 113 directories but no JSON frames. Therefore the
default 10-class configuration excludes the observed directory name
`frog_pose_fg` and produces **1,374 usable clips**, not the approximate 1,202
target. This discrepancy is reported rather than corrected or hidden.

Run the full inspection again with:

```bash
python BiLSTM/scripts/inspect_mmasd.py
```

It writes `artifacts/data_report.json` and `artifacts/class_distribution.csv`.

## Child/person selection limitation

The OpenPose JSON files have `person_id=[-1]` and no child/therapist identity
metadata was found in this modality. The loader therefore uses a deterministic
temporal tracker. It combines confidence/visible-joint coverage, torso and
bounding-box centers, nearest-person continuity, and skeleton-size consistency.
The default has a modest smaller-skeleton prior as a proxy for the child. This
is an implementation heuristic, not a confirmed child label. The strategy is
configurable in YAML, and tracking choices are saved by
`scripts/visualize_sample.py`.

## Leakage prevention

The unit of splitting is a complete clip, never an individual frame. The
default `session_holdout` groups by `subject_id::session_id`, so every clip
from one child and session remains in exactly one split. The fixed seed and
fractions are stored in the config and manifest metadata. Automatic checks
reject duplicate clip IDs, paths, or session groups across splits.

`subject_holdout` is also implemented for later cross-subject experiments.

## Run

Python 3.10/3.11 is preferred. The current development environment also runs
the code on Python 3.13. Install `requirements.txt` in a virtual environment
when needed; the code has NumPy/standard-library fallbacks for pandas and
scikit-learn-dependent convenience functions.

```bash
python BiLSTM/scripts/inspect_mmasd.py

python BiLSTM/scripts/prepare_manifest.py \
  --config BiLSTM/configs/baseline.yaml

python BiLSTM/scripts/visualize_sample.py \
  --clip as_20583_D16_000_i

python BiLSTM/scripts/smoke_test.py \
  --config BiLSTM/configs/debug.yaml

pytest -q BiLSTM/tests
```

Full training is intentionally not started automatically:

```bash
python -m BiLSTM.src.train \
  --config BiLSTM/configs/baseline.yaml

python -m BiLSTM.src.evaluate \
  --config BiLSTM/configs/baseline.yaml \
  --checkpoint BiLSTM/checkpoints/best.pt
```

Evaluation defaults to batch size 1 because full-clip self-attention can use
quadratic memory for long clips. This is separate from the training batch size;
override it with `--batch-size` only if GPU memory allows.

Training selects the best checkpoint using validation macro-F1 only. The test
split is used only by the explicit evaluation command.

## Configuration and reproduction notes

The values in `configs/baseline.yaml` are starting implementation choices when
the exact paper hyperparameters were not recoverable from the local sources.
They are not claimed to be the paper's exact settings.

| Component | Confidence |
|---|---|
| 2D skeleton input | confirmed by local JSON and repository README |
| BODY_25 | confirmed by all local pose arrays |
| MidHip centering | confirmed baseline preprocessing choice |
| Shoulder-width scaling | confirmed baseline preprocessing choice |
| BiLSTM | confirmed target architecture |
| Multi-head attention | confirmed target architecture |
| Hidden size, batch size, learning rate | implementation choice unless independently verified |
| Exact random split | unknown; grouped deterministic protocol is documented |
| Child identity | unavailable in this OpenPose modality; heuristic candidate tracking |

Confidence threshold, missing-joint strategy, smoothing, sequence sampling,
split strategy, model dimensions, optimizer, and early stopping are all in
YAML. The baseline defaults to temporal interpolation, no smoothing, full clip
sequences, Adam, gradient clipping, and validation macro-F1 early stopping.
Near-collapsed shoulder detections use a 5-pixel torso-scale fallback to keep
the normalized features numerically stable.

## Outputs

- `artifacts/data_report.json`: inspection report.
- `artifacts/class_distribution.csv`: raw/usable class counts.
- `artifacts/manifest.csv`: one row per usable clip.
- `artifacts/splits/{train,val,test}.csv`: fixed grouped splits.
- `artifacts/debug/`: raw/tracked/normalized visualizations and tracking JSON.
- `artifacts/results.json`, `per_class_metrics.csv`, `confusion_matrix.png`: evaluation outputs.
- `checkpoints/` and `runs/`: local training outputs, ignored by git.

## Experiments

The existing root-level manifest, split files, `checkpoints/best.pt`, and
`runs/history.json` are preserved as `E0_initial_variable_length`. The paper
reproduction uses separate paths and does not overwrite E0:

- `E1_fixed60_current_architecture`: fixed-60 data with the original model.
- `E2_fixed60_paper_architecture`: fixed-60 data with residual, LayerNorm, and
  the 256-to-64 projection.
- `E3_paper_protocol`: the current paper-protocol-compatible configuration in
  `configs/paper_repro.yaml`.

## Paper reproduction

**Reproduction status:** the published architecture and training settings are
implemented, but the author's exact sample construction, inventory, and split
are unverified. See [the evidence and reproduction status](docs/paper_reproduction_status.md)
and the local [author questions draft](docs/author_reproduction_request.md).
Do not describe this as an exact reproduction of the reported 94.84% result.

Audit the current config and frozen manifests without training or regenerating
splits with `python BiLSTM/scripts/audit_paper_protocol.py` from the repo root.
The pipeline accepts `--reuse-splits`. New checkpoints store config, manifest
and source fingerprints; evaluation checks input settings and recorded CSV
hashes, and exports per-clip predictions and raw/percentage confusion matrices.
Use evaluation's `--output-dir` to preserve previous result files.

The paper-reproduction path keeps the confirmed method components separate
from implementation choices. Its input is 2D BODY_25 x/y data (`60 x 50`),
followed by a two-layer bidirectional LSTM with hidden size 128, four-head
self-attention, residual addition, LayerNorm, temporal mean pooling, Linear
256-to-64, ReLU, Dropout 0.3, and a 10-class classifier. It uses Adam with
learning rate 0.001, batch size 32, CrossEntropyLoss, and validation-accuracy
early stopping with patience 10.

The paper specifies 60-frame sequences but does not explain conversion from
variable-length clips. The default implementation choice is deterministic
uniform temporal sampling for clips longer than 60 and linear temporal
interpolation for clips shorter than 60. Child/person selection, confidence
thresholding, missing-joint handling, shoulder fallbacks, exact split groups,
seed, attention dropout, and internal LSTM dropout are also implementation
choices and are logged in the config/output metadata.

Frog Pose is not present in the paper's 10-class classifier/confusion matrix.
The local MMASD OpenPose tree contains 113 Frog Pose directories but no usable
JSON frames. The paper does not explicitly explain this exclusion, so the
reproduction excludes it without deleting any raw data.

Prepare the paper-compatible session split, reconcile local clip identifiers,
audit tracking, and run the non-training smoke test with:

```bash
python scripts/prepare_manifest.py --config configs/paper_repro.yaml
python scripts/reconcile_clip_counts.py
python scripts/audit_tracking.py
python scripts/smoke_test_paper_repro.py
pytest -q tests
```

After reviewing the preflight outputs, the complete train-and-validation
pipeline can be launched with one command from the `BiLSTM/` directory:

```bash
conda activate mmasd-bilstm
python scripts/run_full_pipeline.py --config configs/paper_repro.yaml
```

Use `--no-train` to run only the preflight stages. The dataset loader caches
processed sequences in memory between epochs when `cache_in_memory: true`.

Paper outputs go to `artifacts/paper_repro/`, split files to
`artifacts/splits/paper_repro/`, checkpoints to `checkpoints/paper_repro/`,
and history to `runs/paper_repro/`. Full 100-epoch training is intentionally
not part of the setup/smoke-test workflow; run it only after reviewing the
reproduction artifacts.
