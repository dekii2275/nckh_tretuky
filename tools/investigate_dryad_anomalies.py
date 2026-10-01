#!/usr/bin/env python3
"""Read-only diagnostics for exact coordinate value 100 in Dryad XLSX files."""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET

from dryad_joint_mapping import RAW_JOINTS


NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
COORDINATE_COLUMNS = 1 + 25 * 3


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper()).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _scalar(cell: ET.Element, shared_strings: list[str]):
    kind = cell.attrib.get("t")
    if kind == "inlineStr":
        return "".join(cell.itertext())
    value = cell.find("main:v", NS)
    if value is None:
        return ""
    text = value.text or ""
    if kind == "s":
        return shared_strings[int(text)]
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def read_sheet(path: Path) -> list[list[object]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(item.itertext()) for item in root.findall("main:si", NS)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[object]] = []
    for row in root.findall("main:sheetData/main:row", NS):
        values: dict[int, object] = {}
        for cell in row.findall("main:c", NS):
            values[_column_index(cell.attrib["r"])] = _scalar(cell, shared_strings)
        width = max(values, default=-1) + 1
        dense = [""] * width
        for index, value in values.items():
            dense[index] = value
        rows.append(dense)
    return rows


def skeleton_files(root: Path) -> list[Path]:
    valid_names = {"8.xlsx", *(f"{number}.xlsx" for number in range(1, 8))}
    files = []
    for path in root.rglob("*.xlsx"):
        if path.name not in valid_names:
            continue
        if path.parent.name == "video" or path.parent.parent.name == "augmentation":
            files.append(path)
    return sorted(files)


def source_metadata(path: Path) -> dict[str, object]:
    marker = "video" if path.parent.name == "video" else "augmentation"
    subject_dir = path.parent.parent if marker == "video" else path.parent.parent.parent
    if "Autism" in path.parts:
        class_name = "ASD"
    elif "Typical" in path.parts:
        class_name = "TD"
    else:
        class_name = "UNKNOWN"
    return {
        "subject_id": subject_dir.name,
        "class": class_name,
        "source": "original" if marker == "video" else "augmentation",
        "augmentation_id": None if marker == "video" else path.parent.name,
    }


def coordinate_triplet(row: list[object], joint_index: int) -> list[float] | None:
    start = 1 + joint_index * 3
    values = row[start:start + 3]
    if len(values) < 3 or not all(isinstance(value, (int, float)) for value in values):
        return None
    return [float(value) for value in values]


def local_report(rows: list[list[object]], frame: int, column: int, joint_index: int) -> dict[str, object]:
    previous = rows[frame - 1][column] if frame > 0 and column < len(rows[frame - 1]) else None
    current = rows[frame][column]
    following = rows[frame + 1][column] if frame + 1 < len(rows) and column < len(rows[frame + 1]) else None
    result: dict[str, object] = {
        "frame_index": frame,
        "raw_column_1_based": column + 1,
        "raw_joint_index": joint_index,
        "raw_joint_name": RAW_JOINTS[joint_index],
        "coordinate_component": (column - 1) % 3,
        "axis": "UNKNOWN",
        "previous": previous,
        "current": current,
        "next": following,
        "abs_current_minus_previous": abs(float(current) - float(previous)) if isinstance(previous, (int, float)) else None,
        "abs_next_minus_current": abs(float(following) - float(current)) if isinstance(following, (int, float)) else None,
    }
    triplet = coordinate_triplet(rows[frame], joint_index)
    if triplet is not None:
        distances = []
        for other_index in range(25):
            if other_index == joint_index:
                continue
            other = coordinate_triplet(rows[frame], other_index)
            if other is not None:
                distances.append(math.dist(triplet, other))
        result["nearest_other_joint_distance"] = min(distances) if distances else None
    sequence_values = []
    for row in rows:
        value = row[column] if column < len(row) else None
        if isinstance(value, (int, float)) and value != 100 and math.isfinite(float(value)):
            sequence_values.append(float(value))
    result["same_coordinate_valid_count"] = len(sequence_values)
    result["same_coordinate_median_excluding_100"] = median(sequence_values) if sequence_values else None
    result["same_coordinate_min_excluding_100"] = min(sequence_values) if sequence_values else None
    result["same_coordinate_max_excluding_100"] = max(sequence_values) if sequence_values else None
    return result


def investigate(root: Path) -> dict[str, object]:
    occurrences = []
    files_with_100 = set()
    subjects_with_100 = set()
    joints_with_100 = set()
    frames_with_100 = set()
    components: dict[str, int] = {"0": 0, "1": 0, "2": 0}
    file_stats = {}
    for path in skeleton_files(root):
        rows = read_sheet(path)
        if not rows:
            continue
        data_rows = rows[1:]
        metadata = source_metadata(path)
        count = 0
        for frame, row in enumerate(data_rows):
            for column in range(1, min(COORDINATE_COLUMNS, len(row))):
                if row[column] != 100:
                    continue
                joint_index = (column - 1) // 3
                component = str((column - 1) % 3)
                report = {
                    **metadata,
                    "file": str(path),
                    **local_report(data_rows, frame, column, joint_index),
                }
                occurrences.append(report)
                count += 1
                files_with_100.add(str(path))
                subjects_with_100.add((metadata["class"], metadata["subject_id"]))
                joints_with_100.add((joint_index, RAW_JOINTS[joint_index]))
                frames_with_100.add((str(path), frame))
                components[component] += 1
        if count:
            file_stats[str(path)] = {"count": count, **metadata, "frames": len(data_rows)}

    return {
        "total_occurrences": len(occurrences),
        "affected_files": sorted(files_with_100),
        "affected_subjects": sorted([f"{cls}/{subject}" for cls, subject in subjects_with_100]),
        "affected_joints": [
            {"raw_joint_index": index, "raw_joint_name": name}
            for index, name in sorted(joints_with_100)
        ],
        "affected_frames": len(frames_with_100),
        "coordinate_components": components,
        "file_stats": file_stats,
        "occurrences": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = investigate(args.root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
