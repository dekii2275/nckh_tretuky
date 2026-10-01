#!/usr/bin/env python3
"""Build subject-level duplicate provenance tables for the Dryad skeletons.

This is a read-only analysis of the released XLSX files.  It writes only
diagnostic CSV metadata; it never rewrites the source workbooks or emits
training tensors.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from audit_dryad_duplicates import scan  # noqa: E402
from investigate_dryad_anomalies import skeleton_files, source_metadata  # noqa: E402


def subject_key(class_name: str, subject_id: str) -> str:
    return f"{class_name}/{subject_id}"


def subject_sort_key(key: str) -> tuple[int, int, str]:
    class_name, subject_id = key.split("/", 1)
    class_rank = {"ASD": 0, "TD": 1}.get(class_name, 2)
    try:
        numeric_id = int(subject_id)
    except ValueError:
        numeric_id = 10**9
    return class_rank, numeric_id, subject_id


class DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def relative_path(path: str, root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path, out_dir: Path) -> dict[str, int]:
    report = scan(root)
    numeric_groups = report["numeric_duplicate_groups"]
    all_subjects = sorted(
        {
            subject_key(
                str(metadata["class"]),
                str(metadata["subject_id"]),
            )
            for path in skeleton_files(root)
            for metadata in [source_metadata(path)]
        },
        key=subject_sort_key,
    )

    dsu = DisjointSet(all_subjects)
    duplicate_rows: list[dict[str, object]] = []
    group_subjects: dict[str, set[str]] = {}
    for group_index, group in enumerate(numeric_groups, start=1):
        group_id = f"NG{group_index:03d}"
        subjects = {
            subject_key(str(record["class"]), str(record["subject_id"]))
            for record in group["files"]
        }
        group_subjects[group_id] = subjects
        ordered_subjects = sorted(subjects, key=subject_sort_key)
        for other in ordered_subjects[1:]:
            dsu.union(ordered_subjects[0], other)
        for record in group["files"]:
            source = str(record["source"])
            duplicate_rows.append(
                {
                    "duplicate_group_id": group_id,
                    "file_path": relative_path(str(record["file"]), root),
                    "subject_id": record["subject_id"],
                    "class": record["class"],
                    "is_original": "1" if source == "original" else "0",
                    "augmentation_id": record["augmentation_id"] or "",
                    "frame_count": record["frame_count"],
                    "byte_hash": record["sha256"],
                    "numeric_hash": record["numeric_digest"],
                }
            )

    components: dict[str, list[str]] = defaultdict(list)
    for subject in all_subjects:
        components[dsu.find(subject)].append(subject)
    ordered_components = sorted(
        components.values(),
        key=lambda members: subject_sort_key(sorted(members, key=subject_sort_key)[0]),
    )
    component_id_by_subject: dict[str, str] = {}
    component_rows: list[dict[str, object]] = []
    for index, members in enumerate(ordered_components, start=1):
        component_id = f"C{index:03d}"
        members = sorted(members, key=subject_sort_key)
        connected_subjects = "|".join(members)
        group_count = sum(
            1 for subjects in group_subjects.values() if subjects.intersection(members)
        )
        for member in members:
            component_id_by_subject[member] = component_id
            class_name, subject_id = member.split("/", 1)
            component_rows.append(
                {
                    "component_id": component_id,
                    "class": class_name,
                    "subject_id": subject_id,
                    "connected_subjects": connected_subjects,
                    "duplicate_group_count": group_count,
                }
            )

    write_csv(
        out_dir / "duplicate_groups.csv",
        [
            "duplicate_group_id",
            "file_path",
            "subject_id",
            "class",
            "is_original",
            "augmentation_id",
            "frame_count",
            "byte_hash",
            "numeric_hash",
        ],
        duplicate_rows,
    )
    write_csv(
        out_dir / "duplicate_subject_components.csv",
        [
            "component_id",
            "class",
            "subject_id",
            "connected_subjects",
            "duplicate_group_count",
        ],
        component_rows,
    )

    multi_subject_components = [members for members in ordered_components if len(members) > 1]
    return {
        "numeric_duplicate_groups": len(numeric_groups),
        "duplicate_group_rows": len(duplicate_rows),
        "subject_components": len(ordered_components),
        "multi_subject_components": len(multi_subject_components),
        "singleton_components": sum(len(members) == 1 for members in ordered_components),
        "largest_component": max(map(len, ordered_components), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    stats = build(args.root, args.out_dir)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
