#!/usr/bin/env python3
"""Read local MMASD+ artifacts; never import their labels as child ground truth."""
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from diagnostics_common import OUT, ROOT, json_dump, preserve_runs, split_records, write_csv


def main():
    preserve_runs()
    base = ROOT.parent
    plus = base / "data_MMASD_plus"
    clips = {r["clip_id"]: r for group in split_records().values() for r in group}
    csvs = sorted(plus.glob("*MEDIAPIPE*/**/*.csv"))
    pattern = re.compile(r"(as|bs|ce|dr|fg|mfs|ms|sac|sq|tr|tw)_\d+_D(?:16|8|1)_\d+_[a-z]")
    inventory, label_counts, schemas = [], Counter(), Counter()
    for path in csvs:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            labels, actions, n = set(), set(), 0
            for row in reader:
                labels.add(row.get("ASD_Label", "")); actions.add(row.get("Action_Label", "")); n += 1
        match = pattern.search(path.stem)
        clip_id = match.group(0) if match else ""
        suffix = path.stem[match.end():] if match else ""
        for label in labels:
            label_counts[label] += 1
        schemas[tuple(fields)] += 1
        inventory.append({"path": str(path), "clip_id_from_filename": clip_id,
                          "matches_frozen_clip": clip_id in clips, "track_and_label_suffix_unverified": suffix,
                          "ASD_Label_values": ";".join(sorted(labels)), "Action_Label_values": ";".join(sorted(actions)),
                          "csv_frames": n, "openpose_frames": int(clips[clip_id]["num_frames"]) if clip_id in clips else None,
                          "frame_count_matches": n == int(clips[clip_id]["num_frames"]) if clip_id in clips else None,
                          "explicit_child_role_mapping_verified": False, "frame_person_alignment_verified": False})
    write_csv(OUT / "external_tracking_asset_alignment.csv", inventory)
    optical = list(plus.glob("*Optical*/**/*.mp4"))
    romp_dirs = sorted(p for p in (plus / "romp_unified").glob("*/*") if p.is_dir())
    npz_count = sum(1 for _ in (plus / "romp_unified").rglob("*.npz"))
    examples = []
    for activity in sorted((plus / "romp_unified").iterdir()):
        folder = next((p for p in sorted(activity.iterdir()) if p.is_dir()), None)
        path = next(folder.glob("*.npz"), None) if folder else None
        if path:
            with np.load(path, allow_pickle=False) as z:
                fields = {}
                for key in z.files:
                    try:
                        value = z[key]; fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                    except ValueError:
                        fields[key] = {"note": "object array not deserialized"}
            examples.append({"path": str(path), "fields": fields})
    archives = []
    for path in (base / "enhanced-mmasd/Sample_Data").glob("*.zip"):
        with zipfile.ZipFile(path) as z:
            archives.append({"path": str(path), "members": len(z.namelist()),
                             "extensions": dict(Counter(Path(n).suffix for n in z.namelist()))})
    exact = {r["clip_id_from_filename"] for r in inventory if r["matches_frozen_clip"]}
    summary = {"mediapipe_csv_files": len(csvs), "optical_flow_mp4_files": len(optical), "romp_track_directories": len(romp_dirs),
               "romp_npz_frames": npz_count, "ASD_Label_file_counts": dict(label_counts),
               "csv_data_row_count_distribution": dict(Counter(r["csv_frames"] for r in inventory)),
               "ASD_Label_nonbinary_values": sorted(set(label_counts) - {"0", "1"}),
               "csv_schemas": [{"columns": list(k), "files": v} for k, v in schemas.items()],
               "frozen_clips_with_filename_match": len(exact), "frozen_clips_total": len(clips),
               "matched_csv_files": sum(r["matches_frozen_clip"] for r in inventory),
               "matched_csv_files_with_equal_frame_count": sum(r["frame_count_matches"] is True for r in inventory),
               "romp_examples": examples, "archives": archives,
               "explicit_binary_label_present": True, "child_therapist_label_semantics_verified": False,
               "original_frame_bbox_track_mapping_found": False,
               "candidate_status": "potentially useful external role/track metadata, not yet aligned or validated child identity"}
    json_dump(summary, OUT / "external_tracking_assets.json")
    lines = ["# Local MMASD+/Enhanced-MMASD tracking assets (read only)", "",
        "No data was downloaded, modified or substituted into the frozen reproduction.", "",
        f"- MediaPipe: {len(csvs)} CSV files; observed schemas and column names are recorded in external_tracking_assets.json. Files include 75 coordinate fields and explicit Action_Label / ASD_Label fields.",
        f"- Optical flow: {len(optical)} person-processed MP4 files.",
        f"- ROMP: {npz_count} NPZ frame outputs in {len(romp_dirs)} processed track directories. Representative schemas from each activity contain cam, center_preds, center_confs, joints, pj2d_org and mesh parameters. No explicit child/therapist field or original crop bounding-box transform was found in these sampled NPZ schemas; not every NPZ payload was opened.",
        "- Enhanced sample archives contain 55 CSV and 55 optical-flow example files, inspected in-place without extraction.", "",
        "## Evidence and meaning", "",
        f"[enhanced-mmasd README]({base / 'enhanced-mmasd/README.md'}) says people are isolated using YOLOv8 and Deep SORT and describes distinguishing therapist/child. [Slide1.JPG]({base / 'enhanced-mmasd/Slide1.JPG'}) illustrates person indices, bounding boxes, cropped videos and a labeling process.",
        f"[proposed_model.py]({base / 'enhanced-mmasd/Code/proposed_model.py'}:49) reads a final filename token as asd_action_label; [line 745]({base / 'enhanced-mmasd/Code/proposed_model.py'}:745) orders its two output labels as Non-ASD, ASD. Actual CSV ASD_Label file counts are {dict(label_counts)}. Thus 0/1 labels exist, but the local data is not consistently binary: values 2, 3 and 6 also occur and are not explained by the inspected two-class code. These are stored dataset labels, not identities inferred by this diagnostic from appearance.",
        "The inspected local documentation does not establish that every ASD_Label=1 track is the target child, every 0 track is the therapist, how those labels were assigned, or their annotation quality. Treat their usefulness for child-vs-other role selection as a hypothesis to validate manually. Do not equate a diagnosis label with a verified role mapping.",
        "The numeric token preceding the binary suffix is track-like, consistent with the README workflow, but no local YOLO/DeepSORT generation script or per-frame track-ID/bounding-box export was found in the inspected repositories/data tree. The tracked/cropped features are generated outputs; annotation provenance and label quality are not established as ground truth.", "",
        "## Alignment to current OpenPose", "",
        f"Filename parsing finds {len(exact)}/{len(clips)} frozen clips represented by {summary['matched_csv_files']} MediaPipe CSVs; {summary['matched_csv_files_with_equal_frame_count']} matched CSVs have the same frame count. CSV data-row counts are {dict(Counter(r['csv_frames'] for r in inventory))}; these fixed-length features do not preserve a documented original-frame mapping. Detailed file/clip candidates are in external_tracking_asset_alignment.csv.",
        "Filename correspondence is only clip-level. Multiple processed tracks can map to one clip; raw OpenPose person-array index is not a DeepSORT ID. Crop coordinate transforms, exact frame/timestamp mapping and identity correspondence are not available in the inspected artifacts. Equal frame counts alone do not verify frame alignment.",
        f"The original [MMASD sample repository]({base / 'MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis'}) contains demo GIFs, OpenPose JSON and an NPZ example, not a verified child-role mapping for all current clips.", "",
        "## Can the model remain 2D-skeleton-only?", "",
        "Yes, in principle: external tracks/role metadata could select a corresponding original OpenPose BODY_25 person while retaining the same 60×50 input. That requires independently validated frame/person alignment and would change the upstream protocol, which must be reported. Feeding MediaPipe 3D, ROMP features or optical flow directly would change this reproduction input and is not proposed here.",
        "", "Decision: useful candidate assets exist, including explicit binary labels, but there is no verified drop-in child identity mapping to current OpenPose frames. Do not replace the tracker automatically."]
    (OUT / "external_tracking_assets.md").write_text("\n".join(lines) + "\n")
    preserve_runs()
    print(json.dumps({k: v for k, v in summary.items() if k not in {"romp_examples", "csv_schemas", "archives"}}, indent=2))


if __name__ == "__main__":
    main()
