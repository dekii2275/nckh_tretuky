"""Dataset discovery without assumptions about raw-data contents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CLIP_NAME_RE = re.compile(
    r"^(?P<prefix>[^_]+)_(?P<subject>[^_]+)_(?P<session>[^_]+)_"
    r"(?P<clip>[^_]+)_(?P<state>[^_]+)$"
)
FRAME_INDEX_RE = re.compile(r"_(?P<index>\d+)_keypoints\.json$")


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    subject_id: str
    session_id: str
    activity: str
    activity_id: int
    num_frames: int
    path: str
    session_group: str
    prefix: str = ""
    clip_index: str = ""
    state: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = FRAME_INDEX_RE.search(path.name)
    return (int(match.group("index")) if match else -1, path.name)


def frame_paths(clip_path: str | Path) -> list[Path]:
    return sorted(Path(clip_path).glob("*_keypoints.json"), key=frame_sort_key)


def parse_clip_name(name: str) -> dict[str, str] | None:
    match = CLIP_NAME_RE.match(name)
    return match.groupdict() if match else None


def discover_classes(skeleton_root: str | Path) -> list[str]:
    root = Path(skeleton_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def discover_clips(
    skeleton_root: str | Path,
    exclude_classes: Iterable[str] = (),
) -> tuple[list[ClipRecord], dict[str, object]]:
    """Return one record per clip directory and discovery diagnostics.

    Empty clip directories are retained in diagnostics but are not silently
    converted into samples by downstream code.
    """
    root = Path(skeleton_root).resolve()
    excluded = set(exclude_classes)
    all_classes = discover_classes(root)
    included_classes = [name for name in all_classes if name not in excluded]
    class_to_id = {name: index for index, name in enumerate(included_classes)}
    records: list[ClipRecord] = []
    empty: list[str] = []
    malformed_names: list[str] = []

    for activity in included_classes:
        activity_dir = root / activity
        for clip_dir in sorted(path for path in activity_dir.iterdir() if path.is_dir()):
            parsed = parse_clip_name(clip_dir.name)
            if parsed is None:
                malformed_names.append(str(clip_dir))
                continue
            count = len(frame_paths(clip_dir))
            if count == 0:
                empty.append(str(clip_dir))
            subject = parsed["subject"]
            session = parsed["session"]
            records.append(
                ClipRecord(
                    clip_id=clip_dir.name,
                    subject_id=subject,
                    session_id=session,
                    activity=activity,
                    activity_id=class_to_id[activity],
                    num_frames=count,
                    path=str(clip_dir),
                    session_group=f"{subject}::{session}",
                    prefix=parsed["prefix"],
                    clip_index=parsed["clip"],
                    state=parsed["state"],
                )
            )

    diagnostics = {
        "root": str(root),
        "all_classes": all_classes,
        "included_classes": included_classes,
        "excluded_classes": sorted(excluded.intersection(all_classes)),
        "class_to_id": class_to_id,
        "empty_clip_count": len(empty),
        "empty_clips": empty,
        "malformed_clip_name_count": len(malformed_names),
        "malformed_clip_names": malformed_names,
    }
    return records, diagnostics

