#!/usr/bin/env python3
"""Read-only duplicate scan for Dryad skeleton XLSX files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from investigate_dryad_anomalies import read_sheet, skeleton_files, source_metadata


def canonical_value(value):
    if isinstance(value, (int, float)):
        return ("number", float(value).hex())
    return ("text", str(value))


def digest_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(len(row)).encode())
        digest.update(b"[")
        for value in row:
            digest.update(repr(canonical_value(value)).encode("utf-8"))
            digest.update(b"|")
        digest.update(b"]")
    return digest.hexdigest()


def scan(root: Path):
    by_bytes = defaultdict(list)
    by_numeric = defaultdict(list)
    records = {}
    for path in skeleton_files(root):
        data = path.read_bytes()
        rows = read_sheet(path)
        metadata = source_metadata(path)
        record = {
            "file": str(path),
            **metadata,
            "sha256": hashlib.sha256(data).hexdigest(),
            "numeric_digest": digest_rows(rows),
            "frame_count": max(0, len(rows) - 1),
        }
        records[str(path)] = record
        by_bytes[record["sha256"]].append(str(path))
        by_numeric[record["numeric_digest"]].append(str(path))

    def groups(index):
        result = []
        for key, paths in sorted(index.items()):
            if len(paths) > 1:
                result.append({"digest": key, "files": [records[path] for path in sorted(paths)]})
        return result

    return {
        "file_count": len(records),
        "byte_duplicate_groups": groups(by_bytes),
        "numeric_duplicate_groups": groups(by_numeric),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(scan(args.root), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
