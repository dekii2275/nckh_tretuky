#!/usr/bin/env python3
"""Save raw, tracked, and normalized skeleton views for one clip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.child_tracking import select_child_track
from src.data.discover import discover_clips
from src.data.openpose_parser import parse_clip
from src.data.preprocessing import preprocess_clip
from src.utils import cfg_path, ensure_dir, load_config


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18), (14, 21), (11, 24),
    (19, 20), (22, 23),
]


def draw_skeleton(axis, people, selected=None, title="", normalized=False):
    for person in people:
        coords = person[:, :2]
        axis.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.18, color="tab:blue")
        for first, second in EDGES:
            axis.plot(
                [coords[first, 0], coords[second, 0]],
                [coords[first, 1], coords[second, 1]],
                color="tab:blue",
                alpha=0.16,
                linewidth=0.6,
            )
    if selected is not None:
        coords = selected[:, :2] if selected.shape[-1] == 3 else selected
        axis.scatter(coords[:, 0], coords[:, 1], s=12, color="tab:red", label="tracked candidate")
        for first, second in EDGES:
            axis.plot(
                [coords[first, 0], coords[second, 0]],
                [coords[first, 1], coords[second, 1]],
                color="tab:red",
                linewidth=1.2,
            )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    axis.invert_yaxis() if not normalized else None
    axis.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True, help="Exact clip directory name")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/debug")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = cfg_path(config, "data", "root", default="data_MMASD")
    skeleton_dir = Path(str(config.get("data", {}).get("skeleton_dir", "2D_skeleton/2D_openpose/output")))
    skeleton_root = skeleton_dir if skeleton_dir.is_absolute() else data_root / skeleton_dir
    records, _ = discover_clips(skeleton_root)
    matches = [record for record in records if record.clip_id == args.clip]
    if not matches:
        raise FileNotFoundError(f"Clip not found: {args.clip}")
    record = matches[0]
    frames, _ = parse_clip(record.path, strict=True)
    tracking = select_child_track(
        frames,
        confidence_threshold=float(config.get("data", {}).get("confidence_threshold", 0.1)),
        strategy=str(config.get("data", {}).get("tracking_strategy", "temporal_nearest")),
        prefer_smaller=bool(config.get("data", {}).get("prefer_smaller_person", True)),
    )
    processed = preprocess_clip(
        tracking.selected_people,
        confidence_threshold=float(config.get("data", {}).get("confidence_threshold", 0.1)),
        missing_joint_strategy=str(config.get("data", {}).get("missing_joint_strategy", "interpolate")),
        smoothing=bool(config.get("data", {}).get("smoothing", False)),
    )
    indices = sorted({0, len(frames) // 2, len(frames) - 1})
    figure, axes = plt.subplots(2, len(indices), figsize=(5 * len(indices), 8), squeeze=False)
    for column, frame_index in enumerate(indices):
        selected = tracking.selected_people[frame_index]
        draw_skeleton(
            axes[0, column],
            frames[frame_index],
            selected=selected,
            title=f"Raw + tracked, frame {frame_index}",
        )
        draw_skeleton(
            axes[1, column],
            [processed.coordinates[frame_index]],
            selected=processed.coordinates[frame_index],
            title=f"Normalized, frame {frame_index}",
            normalized=True,
        )
    figure.suptitle(f"{record.clip_id} | activity={record.activity}")
    figure.tight_layout()
    output_dir = ensure_dir(args.output_dir)
    output_path = output_dir / f"{record.clip_id}_debug.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    debug = tracking.debug_dict(record.clip_id)
    debug["preprocessing"] = processed.metadata
    debug["frame_indices_visualized"] = indices
    with (output_dir / f"{record.clip_id}_tracking.json").open("w", encoding="utf-8") as handle:
        json.dump(debug, handle, indent=2)
    print(f"Saved {output_path}")
    print(f"Saved {output_dir / f'{record.clip_id}_tracking.json'}")


if __name__ == "__main__":
    main()
