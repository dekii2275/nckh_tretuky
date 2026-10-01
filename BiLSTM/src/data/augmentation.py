"""Skeleton-level data augmentation for 2D pose sequences.

Applied AFTER normalization (center at MidHip, scale by shoulder width).
All augmentations preserve the [T, J, 2] format.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .openpose_parser import BODY_25_INDICES

# OpenPose BODY_25 left-right swap pairs
BODY25_SWAP_PAIRS = [
    (BODY_25_INDICES["RShoulder"], BODY_25_INDICES["LShoulder"]),
    (BODY_25_INDICES["RElbow"], BODY_25_INDICES["LElbow"]),
    (BODY_25_INDICES["RWrist"], BODY_25_INDICES["LWrist"]),
    (BODY_25_INDICES["RHip"], BODY_25_INDICES["LHip"]),
    (BODY_25_INDICES["RKnee"], BODY_25_INDICES["LKnee"]),
    (BODY_25_INDICES["RAnkle"], BODY_25_INDICES["LAnkle"]),
    (BODY_25_INDICES["REye"], BODY_25_INDICES["LEye"]),
    (BODY_25_INDICES["REar"], BODY_25_INDICES["LEar"]),
    (BODY_25_INDICES["RBigToe"], BODY_25_INDICES["LBigToe"]),
    (BODY_25_INDICES["RSmallToe"], BODY_25_INDICES["LSmallToe"]),
    (BODY_25_INDICES["RHeel"], BODY_25_INDICES["LHeel"]),
]


class SkeletonAugmentation:
    """Configurable skeleton augmentation pipeline.

    All operations work on normalized [T, J, 2] arrays and are applied
    only during training.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.flip_prob = float(config.get("flip_prob", 0.5))
        self.noise_std = float(config.get("noise_std", 0.02))
        self.rotation_range = float(config.get("rotation_range_deg", 10.0))
        self.joint_dropout_prob = float(config.get("joint_dropout_prob", 0.1))
        self.max_dropout_joints = int(config.get("max_dropout_joints", 3))
        self.time_warp_range = tuple(config.get("time_warp_range", [0.9, 1.1]))
        self.random_shift_frames = int(config.get("random_shift_frames", 5))

    def __call__(self, sequence: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Apply random augmentations to a [T, J, 2] sequence."""
        if not self.enabled:
            return sequence

        rng = rng or np.random.default_rng()
        result = sequence.copy()

        # 1. Horizontal flip
        if self.flip_prob > 0 and rng.random() < self.flip_prob:
            result = self._horizontal_flip(result)

        # 2. Small random rotation
        if self.rotation_range > 0:
            angle = rng.uniform(-self.rotation_range, self.rotation_range)
            result = self._rotate(result, angle)

        # 3. Gaussian noise
        if self.noise_std > 0:
            noise = rng.standard_normal(result.shape).astype(np.float32) * self.noise_std
            result = result + noise

        # 4. Random joint dropout
        if self.joint_dropout_prob > 0:
            result = self._joint_dropout(result, rng)

        # 5. Temporal shift (only meaningful for windowed data)
        if self.random_shift_frames > 0 and result.shape[0] > 2 * self.random_shift_frames:
            shift = rng.integers(-self.random_shift_frames, self.random_shift_frames + 1)
            if shift != 0:
                result = np.roll(result, shift, axis=0)
                if shift > 0:
                    result[:shift] = result[shift]
                else:
                    result[shift:] = result[shift - 1]

        return result.astype(np.float32)

    def _horizontal_flip(self, sequence: np.ndarray) -> np.ndarray:
        """Flip x coordinates and swap left/right joints."""
        result = sequence.copy()
        result[:, :, 0] *= -1
        for left, right in BODY25_SWAP_PAIRS:
            result[:, [left, right]] = result[:, [right, left]]
        return result

    def _rotate(self, sequence: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate all joints by a small angle around origin (MidHip = 0,0)."""
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        # [T, J, 2] @ [2, 2] -> [T, J, 2]
        return np.einsum("tji,ki->tjk", sequence, rotation)

    def _joint_dropout(self, sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Zero out a random subset of joints."""
        n_joints = sequence.shape[1]
        n_drop = rng.integers(0, min(self.max_dropout_joints, n_joints) + 1)
        if n_drop == 0:
            return sequence
        drop_indices = rng.choice(n_joints, size=n_drop, replace=False)
        result = sequence.copy()
        result[:, drop_indices, :] = 0.0
        return result
