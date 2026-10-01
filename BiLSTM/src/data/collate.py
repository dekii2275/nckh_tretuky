"""Variable-length sequence collation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


def skeleton_collate(batch: Sequence[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    sequences = [item["skeleton"] for item in batch]
    if not all(isinstance(value, torch.Tensor) for value in sequences):
        raise TypeError("Each sample must contain a Tensor under 'skeleton'")
    lengths = torch.tensor([value.shape[0] for value in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    max_length = padded.shape[1]
    padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
    return {
        "skeleton": padded,
        "sequence": padded,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "clip_ids": [str(item["clip_id"]) for item in batch],
        "paths": [str(item["path"]) for item in batch],
        "metadata": [{key: value for key, value in item.items()
                      if key not in {"skeleton", "tracking", "preprocessing", "valid_mask"}}
                     for item in batch],
    }
