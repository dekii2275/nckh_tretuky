"""Small, explicit tensor policies frozen by Dryad/MSG3D DATA_CONTRACT_v1."""

from __future__ import annotations

import numpy as np


CONTRACT_VERSION = "DATA_CONTRACT_v1"
TARGET_T = 60
NUM_CHANNELS = 3
NUM_JOINTS = 25
NUM_PERSON = 2
SENTINEL_VALUE = 100.0


def right_zero_pad(sequence: np.ndarray, target_t: int = TARGET_T) -> np.ndarray:
    """Return [C,target_t,V] without inventing frames or editing the input."""

    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != NUM_CHANNELS or array.shape[2] != NUM_JOINTS:
        raise ValueError("expected a [3,T,25] skeleton")
    if array.shape[1] > target_t:
        raise ValueError(f"T_raw={array.shape[1]} exceeds target_t={target_t}")
    padded = np.zeros((NUM_CHANNELS, target_t, NUM_JOINTS), dtype=np.float32)
    padded[:, : array.shape[1], :] = array
    return padded


def stack_zero_person(sequence: np.ndarray, target_t: int = TARGET_T) -> np.ndarray:
    """Return model input [C,target_t,V,M] with an empty second person slot."""

    real = right_zero_pad(sequence, target_t=target_t)
    empty = np.zeros_like(real)
    return np.stack((real, empty), axis=-1)


def sentinel_mask(sequence: np.ndarray, value: float = SENTINEL_VALUE) -> np.ndarray:
    """Detect the contract sentinel exactly; no repair is performed here."""

    return np.asarray(sequence) == value
