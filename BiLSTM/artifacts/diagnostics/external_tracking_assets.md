# Local MMASD+/Enhanced-MMASD tracking assets (read only)

No data was downloaded, modified or substituted into the frozen reproduction.

- MediaPipe: 3276 CSV files; observed schemas and column names are recorded in external_tracking_assets.json. Files include 75 coordinate fields and explicit Action_Label / ASD_Label fields.
- Optical flow: 3267 person-processed MP4 files.
- ROMP: 642636 NPZ frame outputs in 3251 processed track directories. Representative schemas from each activity contain cam, center_preds, center_confs, joints, pj2d_org and mesh parameters. No explicit child/therapist field or original crop bounding-box transform was found in these sampled NPZ schemas; not every NPZ payload was opened.
- Enhanced sample archives contain 55 CSV and 55 optical-flow example files, inspected in-place without extraction.

## Evidence and meaning

[enhanced-mmasd README](/home/dekii2275/nckh_tretuky/enhanced-mmasd/README.md) says people are isolated using YOLOv8 and Deep SORT and describes distinguishing therapist/child. [Slide1.JPG](/home/dekii2275/nckh_tretuky/enhanced-mmasd/Slide1.JPG) illustrates person indices, bounding boxes, cropped videos and a labeling process.
[proposed_model.py](/home/dekii2275/nckh_tretuky/enhanced-mmasd/Code/proposed_model.py:49) reads a final filename token as asd_action_label; [line 745](/home/dekii2275/nckh_tretuky/enhanced-mmasd/Code/proposed_model.py:745) orders its two output labels as Non-ASD, ASD. Actual CSV ASD_Label file counts are {'0': 1315, '1': 1258, '6': 1, '2': 697, '3': 5}. Thus 0/1 labels exist, but the local data is not consistently binary: values 2, 3 and 6 also occur and are not explained by the inspected two-class code. These are stored dataset labels, not identities inferred by this diagnostic from appearance.
The inspected local documentation does not establish that every ASD_Label=1 track is the target child, every 0 track is the therapist, how those labels were assigned, or their annotation quality. Treat their usefulness for child-vs-other role selection as a hypothesis to validate manually. Do not equate a diagnosis label with a verified role mapping.
The numeric token preceding the binary suffix is track-like, consistent with the README workflow, but no local YOLO/DeepSORT generation script or per-frame track-ID/bounding-box export was found in the inspected repositories/data tree. The tracked/cropped features are generated outputs; annotation provenance and label quality are not established as ground truth.

## Alignment to current OpenPose

Filename parsing finds 1069/1374 frozen clips represented by 2753 MediaPipe CSVs; 0 matched CSVs have the same frame count. CSV data-row counts are {179: 3276}; these fixed-length features do not preserve a documented original-frame mapping. Detailed file/clip candidates are in external_tracking_asset_alignment.csv.
Filename correspondence is only clip-level. Multiple processed tracks can map to one clip; raw OpenPose person-array index is not a DeepSORT ID. Crop coordinate transforms, exact frame/timestamp mapping and identity correspondence are not available in the inspected artifacts. Equal frame counts alone do not verify frame alignment.
The original [MMASD sample repository](/home/dekii2275/nckh_tretuky/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis) contains demo GIFs, OpenPose JSON and an NPZ example, not a verified child-role mapping for all current clips.

## Can the model remain 2D-skeleton-only?

Yes, in principle: external tracks/role metadata could select a corresponding original OpenPose BODY_25 person while retaining the same 60×50 input. That requires independently validated frame/person alignment and would change the upstream protocol, which must be reported. Feeding MediaPipe 3D, ROMP features or optical flow directly would change this reproduction input and is not proposed here.

Decision: useful candidate assets exist, including explicit binary labels, but there is no verified drop-in child identity mapping to current OpenPose frames. Do not replace the tracker automatically.
