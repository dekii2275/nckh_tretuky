"""Loader-compatible serialization and checksum checks."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "dryad_msg3d"


def test_checksum_manifest_covers_existing_files():
    checksums = json.loads((ARTIFACT_ROOT / "checksums.json").read_text(encoding="utf-8"))
    assert len(checksums) >= 170
    for relative, expected in checksums.items():
        path = ARTIFACT_ROOT / relative
        assert path.is_file()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected


def test_loader_pickle_structure_matches_current_dataset_expectation():
    loader_root = ARTIFACT_ROOT / "loader" / "random_reproduction_like"
    data = np.load(loader_root / "train_data_fold_00.npy", mmap_mode="r")
    with (loader_root / "train_labels_fold_00.pkl").open("rb") as handle:
        labels, sample_paths = pickle.load(handle)
    assert data.shape == (720, 3, 60, 25, 2)
    assert data.dtype == np.float32
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert labels.shape == (720,)
    assert isinstance(sample_paths, list)
    assert len(sample_paths) == 720
    assert set(labels.tolist()) == {0, 1}
