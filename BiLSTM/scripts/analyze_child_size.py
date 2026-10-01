#!/usr/bin/env python3
"""Measure size only for explicitly human-identified same-frame role pairs."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnostics_common import OUT, REVIEW, json_dump, preserve_runs, read_labels, read_manifest, write_csv
from src.data.openpose_parser import parse_clip

FIELDS = "clip_id subject_id session_id activity frame_index child_person_index therapist_person_index review_label status child_bbox_area therapist_bbox_area child_skeleton_height therapist_skeleton_height area_ratio height_ratio area_comparison height_comparison notes".split()


def size(person):
    valid = np.isfinite(person).all(1) & (person[:, 2] >= 0.1) & np.any(person[:, :2] != 0, axis=1)
    if valid.sum() < 2:
        return None, None
    extent = np.ptp(person[valid, :2], axis=0)
    return float(extent.prod()), float(extent[1])


def comparison(ratio):
    return "ambiguous" if ratio is None or 0.95 <= ratio <= 1.05 else "child_smaller" if ratio < 0.95 else "child_larger"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=REVIEW / "manual_person_pairs.csv")
    parser.add_argument("--labels", type=Path, default=REVIEW / "manual_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    preserve_runs()
    labels = {r["clip_id"]: r for r in read_labels(args.labels)}
    records = {r["clip_id"]: r for r in read_manifest(REVIEW / "review_manifest.csv")}
    pairs = read_manifest(args.pairs) if args.pairs.exists() else []
    rows, cache, seen = [], {}, set()
    for pair in pairs:
        key = pair["clip_id"]
        if key not in records:
            raise ValueError(f"Unknown review clip {key}")
        frame = int(pair["frame_index"])
        if (key, frame) in seen:
            raise ValueError("Duplicate role pair at same clip/frame")
        seen.add((key, frame))
        child, therapist = int(pair["child_person_index"]), int(pair["therapist_person_index"])
        if key not in cache:
            cache[key], _ = parse_clip(records[key]["path"], strict=True)
        frames = cache[key]
        if not 0 <= frame < len(frames) or child == therapist or min(child, therapist) < 0 or max(child, therapist) >= len(frames[frame]):
            raise ValueError(f"Invalid frame/person indices: {key}")
        if pair["roles_confident"] not in {"0", "1"} or pair["both_visible"] not in {"0", "1"}:
            raise ValueError("roles_confident/both_visible must be human-entered 0 or 1")
        row = {field: None for field in FIELDS}
        row.update({k: records[key][k] for k in ("clip_id", "subject_id", "session_id", "activity")})
        row.update(frame_index=frame, child_person_index=child, therapist_person_index=therapist,
                   review_label=labels.get(key, {}).get("review_label", ""), notes=pair.get("notes", ""),
                   status="human_roles_not_confirmed", area_comparison="ambiguous", height_comparison="ambiguous")
        if row["review_label"] and pair["roles_confident"] == "1" and pair["both_visible"] == "1":
            ca, ch = size(frames[frame][child]); ta, th = size(frames[frame][therapist])
            ar = ca / ta if ca is not None and ta is not None and ta > 0 and ca > 0 else None
            hr = ch / th if ch is not None and th is not None and th > 0 and ch > 0 else None
            row.update(status="measured_human_confirmed_roles", child_bbox_area=ca, therapist_bbox_area=ta,
                       child_skeleton_height=ch, therapist_skeleton_height=th, area_ratio=ar, height_ratio=hr,
                       area_comparison=comparison(ar), height_comparison=comparison(hr))
        rows.append(row)
    write_csv(args.output_dir / "child_size_assumption.csv", rows, FIELDS)
    by_clip = defaultdict(list)
    for row in rows:
        if row["review_label"]:
            by_clip[row["clip_id"]].append(row)
    categories = {}
    for measure in ("area", "height"):
        counts = {"child_smaller": 0, "child_larger": 0, "ambiguous": 0}
        for group in by_clip.values():
            values = [r[f"{measure}_ratio"] for r in group if r[f"{measure}_ratio"] is not None]
            category = comparison(float(np.median(values)) if values else None)
            counts[category] += 1
        categories[measure] = {"clip_counts": counts, "clip_percentages": {k: 100 * v / len(by_clip) if by_clip else None for k, v in counts.items()}}
    summary = {"status": "awaiting_human_role_pairs" if not rows else "human_pair_measurements_available",
               "annotated_frame_pairs": len(rows), "reviewed_clips_with_pairs": len(by_clip),
               "confident_measured_pairs": sum(r["status"] == "measured_human_confirmed_roles" for r in rows),
               "results": categories, "primary_unit": "one reviewed clip: median available confirmed frame-pair ratio; no available measurement counts as ambiguous",
               "size_definition": "bbox area and vertical extent of original OpenPose joints with confidence>=0.1 and nonzero finite coordinates",
               "ambiguity_rule": "predeclared +/-5% ratio band, invalid bbox/height, or human role not confirmed; no human pairs means percentages unavailable, not zero",
               "limitations": "Frame/clip selection is manual and nonrandom; occlusion distorts bbox size. Existing initial score uses confidence*log1p(valid_count)/sqrt(diagonal), so bbox area/height audit tests the size-prior assumption, not the entire selection score.",
               "tracker_changed": False, "oracle_test_performed": False}
    json_dump(summary, args.output_dir / "child_size_assumption_summary.json")
    preserve_runs()
    print(summary["status"], "confirmed pairs:", summary["confident_measured_pairs"])


if __name__ == "__main__":
    main()
