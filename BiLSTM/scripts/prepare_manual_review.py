#!/usr/bin/env python3
"""Reuse existing 120 raw-frame payloads; add human labels and role-pair entry."""
import json
import re
import hashlib

from diagnostics_common import REVIEW, ROOT, json_dump, preserve_runs, read_labels, read_manifest, write_csv

LABEL_FIELDS = "clip_id subject_id session_id activity num_frames max_people review_label multiple_people partial_occlusion identity_switch_visible notes child_leaves_frame therapist_crosses_child severe_keypoint_failure tiny_skeleton large_camera_motion".split()
PAIR_FIELDS = "clip_id frame_index child_person_index therapist_person_index roles_confident both_visible notes".split()


def main():
    preserve_runs()
    records = read_manifest(REVIEW / "review_manifest.csv")
    old = {r["clip_id"]: r for r in read_labels()}
    labels = []
    for record in records:
        row = {f: record.get(f, "") if f in LABEL_FIELDS[:6] else "" for f in LABEL_FIELDS}
        row.update(old.get(record["clip_id"], {}))
        labels.append(row)
    write_csv(REVIEW / "manual_labels.csv", labels, LABEL_FIELDS)
    if not (REVIEW / "manual_person_pairs.csv").exists():
        write_csv(REVIEW / "manual_person_pairs.csv", [], PAIR_FIELDS)
    person_pairs = read_manifest(REVIEW / "manual_person_pairs.csv")
    revision = hashlib.sha256((REVIEW / "manual_labels.csv").read_bytes() + (REVIEW / "manual_person_pairs.csv").read_bytes()).hexdigest()[:12]
    template = (ROOT / "scripts/tracking_review_page.html").read_text()
    for record, label in zip(records, labels, strict=True):
        path = REVIEW / (record["clip_id"] + ".html")
        match = re.search(r'<script id="data" type="application/json">(.*?)</script>', path.read_text(), re.S)
        if not match:
            raise ValueError(f"Missing original raw-frame payload: {path}")
        payload = json.loads(match.group(1))
        payload["initial_label"] = label
        payload["label_fields"] = LABEL_FIELDS
        payload["pair_fields"] = PAIR_FIELDS
        payload["initial_pairs"] = [r for r in person_pairs if r["clip_id"] == record["clip_id"]]
        payload["revision"] = revision
        path.write_text(template.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")))
    index = (ROOT / "scripts/tracking_review_index.html").read_text()
    payload = {"rows": labels, "label_fields": LABEL_FIELDS, "pair_fields": PAIR_FIELDS, "person_pairs": person_pairs, "revision": revision}
    (REVIEW / "index.html").write_text(index.replace("__DATA__", json.dumps(payload).replace("<", "\\u003c")))
    json_dump({"labels": {"A": "clearly tracks child correctly", "B": "clearly tracks therapist/wrong person",
                          "C": "starts correctly but switches identity", "D": "uncertain/impossible to determine"},
               "blank": "not reviewed (not D)", "flags": "blank=not assessed; 1=yes; 0=no",
               "fields": LABEL_FIELDS, "pair_fields": PAIR_FIELDS,
               "frame_indices": "zero based, raw OpenPose array index in this exact frame; indices may reorder",
               "pair_annotation": "Human explicitly identifies child and therapist at the same frame; no role inferred from size or ASD_Label",
               "workflow": "Open index in browser. Labels are saved locally where available. Export all CSVs from index; copy downloaded manual_labels.csv and manual_person_pairs.csv into this folder, then run analyze_tracking_labels.py and analyze_child_size.py. Browser storage alone does not modify files.",
               "legacy_reviews": "Prior browser review namespace is preserved and not automatically mapped into A/B/C/D.",
               "oracle": "Frame pairs support size auditing only; they do not define a complete oracle child track. No oracle inference/training is performed."},
              REVIEW / "label_schema.json")
    preserve_runs()
    print(f"Prepared {len(labels)} human label rows; {sum(bool(r['review_label']) for r in labels)} labeled.")


if __name__ == "__main__":
    main()
