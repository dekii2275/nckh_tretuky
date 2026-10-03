#!/usr/bin/env python3
"""Render two representative skeleton frames from MMASD, DRYAD, and DREAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MMASD = ROOT / "data_MMASD/2D_skeleton/2D_openpose/output"
DRYAD = ROOT / "Data_DRYAD/doi_10_5061_dryad_s7h44j150__v20200911/Dataset-2/Dataset"
DREAM = ROOT / "data_Dream/DREAMdataset/User 33"

OPENPOSE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18),
    (11, 22), (22, 23), (23, 24), (14, 19), (19, 20), (20, 21),
]
KINECT_NAMES = [
    "SpineBase", "Midspain", "SpineShoulder", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft", "HandTipLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight", "HandTipRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
]
KINECT_LINKS = [
    ("SpineBase", "Midspain"), ("Midspain", "SpineShoulder"),
    ("SpineShoulder", "Neck"), ("Neck", "Head"),
    ("SpineShoulder", "ShoulderLeft"), ("ShoulderLeft", "ElbowLeft"),
    ("ElbowLeft", "WristLeft"), ("WristLeft", "HandLeft"),
    ("HandLeft", "HandTipLeft"),
    ("SpineShoulder", "ShoulderRight"), ("ShoulderRight", "ElbowRight"),
    ("ElbowRight", "WristRight"), ("WristRight", "HandRight"),
    ("HandRight", "HandTipRight"),
    ("SpineBase", "HipLeft"), ("HipLeft", "KneeLeft"),
    ("KneeLeft", "AnkleLeft"), ("AnkleLeft", "FootLeft"),
    ("SpineBase", "HipRight"), ("HipRight", "KneeRight"),
    ("KneeRight", "AnkleRight"), ("AnkleRight", "FootRight"),
]
DREAM_NAMES = [
    "head", "sholder_center", "sholder_left", "elbow_left", "wrist_left",
    "hand_left", "sholder_right", "elbow_right", "wrist_right", "hand_right",
]
DREAM_LINKS = [
    ("head", "sholder_center"),
    ("sholder_center", "sholder_left"), ("sholder_left", "elbow_left"),
    ("elbow_left", "wrist_left"), ("wrist_left", "hand_left"),
    ("sholder_center", "sholder_right"), ("sholder_right", "elbow_right"),
    ("elbow_right", "wrist_right"), ("wrist_right", "hand_right"),
]


def openpose_frame(path: Path) -> tuple[np.ndarray, list[tuple[int, int]], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = [np.asarray(p["pose_keypoints_2d"], dtype=float).reshape(25, 3)
             for p in data["people"] if len(p.get("pose_keypoints_2d", [])) == 75]
    if not poses:
        raise ValueError(f"No BODY_25 pose in {path}")
    scores = [float(np.where(p[:, 2] >= 0.25, p[:, 2], 0).sum()) for p in poses]
    person_index = int(np.argmax(scores))
    pose = poses[person_index]
    xy = pose[:, :2].copy()
    xy[(pose[:, 2] < 0.25) | ((xy[:, 0] == 0) & (xy[:, 1] == 0))] = np.nan
    xy[:, 1] *= -1  # OpenPose image y points downward.
    return xy, OPENPOSE_BONES, {"person_index": person_index, "joint_count": 25}


def dryad_frame(path: Path, frame: int) -> tuple[np.ndarray, list[tuple[int, int]], dict]:
    table = pd.read_excel(path)
    if not 0 <= frame < len(table):
        raise ValueError(f"Frame {frame} outside {path} ({len(table)} frames)")
    points = []
    for name in KINECT_NAMES:
        column = table.columns.get_loc(name)
        xyz = table.iloc[frame, column:column + 3].to_numpy(dtype=float)
        xy = xyz[:2].copy()  # Front view of original Kinect 3D coordinates.
        if not np.isfinite(xyz).all() or np.any(xyz == 100):
            xy[:] = np.nan
        points.append(xy)
    names = {name: i for i, name in enumerate(KINECT_NAMES)}
    links = [(names[a], names[b]) for a, b in KINECT_LINKS]
    return np.asarray(points), links, {"joint_count": 23, "frame": frame}


def dream_frame(path: Path, frame: int) -> tuple[np.ndarray, list[tuple[int, int]], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    joints = data["skeleton"]
    points = []
    for name in DREAM_NAMES:
        signal = joints[name]
        x, y, confidence = (signal[key][frame] for key in ("x", "y", "confidence"))
        xy = [float(x), float(y)] if x is not None and y is not None and (confidence or 0) > 0 else [np.nan, np.nan]
        points.append(xy)
    names = {name: i for i, name in enumerate(DREAM_NAMES)}
    links = [(names[a], names[b]) for a, b in DREAM_LINKS]
    return np.asarray(points), links, {
        "joint_count": 10, "frame": frame, "task": data["task"]["ability"],
        "condition": data["condition"],
    }


def fit_pose(points: np.ndarray) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1)
    if valid.sum() < 5:
        raise ValueError("Fewer than five valid joints")
    result = points.copy()
    lo = np.nanmin(result, axis=0)
    hi = np.nanmax(result, axis=0)
    width, height = hi - lo
    scale = min(1.25 / max(width, 1e-9), 1.16 / max(height, 1e-9))
    result[:, 0] = (result[:, 0] - (lo[0] + hi[0]) / 2) * scale
    result[:, 1] = (result[:, 1] - lo[1]) * scale + 0.18 + (1.16 - height * scale) / 2
    return result


def draw(ax, points: np.ndarray, links: list[tuple[int, int]], color: str, label: str, detail: str) -> None:
    points = fit_pose(points)
    ax.set_facecolor("#ffffff")
    for a, b in links:
        if np.isfinite(points[[a, b]]).all():
            ax.plot(points[[a, b], 0], points[[a, b], 1], color=color,
                    linewidth=3.4, solid_capstyle="round", zorder=2)
    valid = np.isfinite(points).all(axis=1)
    ax.scatter(points[valid, 0], points[valid, 1], s=25, color="#ffffff",
               edgecolors=color, linewidths=2, zorder=3)
    ax.set(xlim=(-0.8, 0.8), ylim=(-0.09, 1.56), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dbe2ea")
        spine.set_linewidth(1.3)
    ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=12, color="#152338", fontweight="bold")
    ax.text(0.04, 0.055, detail, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8.5, color="#657486")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/visualizations/three_datasets_2x3.png")
    args = parser.parse_args()
    samples = [
        ("MMASD", "Arm swing", "OpenPose 2D · frame 60", "#127c91", "openpose",
         MMASD / "arm_swing_as/as_20583_D1_000_y/as_20583_D1_000_y_000000000060_keypoints.json", None),
        ("DRYAD", "Typical gait", "Kinect 3D → XY · frame 17", "#b5681b", "dryad",
         DRYAD / "Typical/1/video/8.xlsx", 17),
        ("DREAM", "Joint attention", "10 upper-body joints · XY · frame 1547", "#7856ac", "dream",
         DREAM / "User 33_12_intervention 2_20170623_093149.724000.json", 1547),
        ("MMASD", "Tree pose", "OpenPose 2D · frame 87", "#127c91", "openpose",
         MMASD / "tree_pose_tr/tr_41004_D1_000_y/tr_41004_D1_000_y_000000000087_keypoints.json", None),
        ("DRYAD", "ASD gait", "Kinect 3D → XY · frame 17", "#b5681b", "dryad",
         DRYAD / "Autism/children with ASD/1/video/8.xlsx", 17),
        ("DREAM", "Imitation", "10 upper-body joints · XY · frame 5494", "#7856ac", "dream",
         DREAM / "User 33_10_intervention 1_20170623_090815.158000.json", 5494),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "#f3f6fa"})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.0), facecolor="#f3f6fa")
    manifest = []
    for ax, (dataset, label, detail, color, kind, path, frame) in zip(axes.flat, samples):
        if not path.is_file():
            raise FileNotFoundError(path)
        if kind == "openpose":
            points, links, info = openpose_frame(path)
        elif kind == "dryad":
            points, links, info = dryad_frame(path, frame)
        else:
            points, links, info = dream_frame(path, frame)
        draw(ax, points, links, color, label, detail)
        manifest.append({"dataset": dataset, "label": label, "source": str(path.relative_to(ROOT)), **info})
    for ax, title, color in zip(axes[0], ("MMASD", "DRYAD", "DREAM"), ("#127c91", "#b5681b", "#7856ac")):
        ax.set_title(title, color=color, fontsize=17, fontweight="bold", pad=15)
    fig.suptitle("Ba bộ dữ liệu · 2 mẫu mỗi bộ", x=0.5, y=0.985,
                 fontsize=19, fontweight="bold", color="#152338")
    fig.text(0.5, 0.027,
             "Mỗi ô là một frame, co giãn riêng; DREAM chỉ có 10 khớp thân trên, không có hông và chân.",
             ha="center", fontsize=10, color="#657486")
    fig.subplots_adjust(left=0.045, right=0.955, top=0.89, bottom=0.075, wspace=0.105, hspace=0.18)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
