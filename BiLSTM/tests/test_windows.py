import numpy as np
import pytest
import torch

from src.data.dataset import SkeletonWindowDataset, window_metadata
from src.window_study import aggregate_clip_logits


def record(clip_id="a", frames=150):
    return {"clip_id": clip_id, "subject_id": "s", "session_id": "D1", "activity": "action",
            "activity_id": 0, "num_frames": frames, "path": clip_id}


@pytest.mark.parametrize("stride,starts", [(60, [0, 60]), (30, [0, 30, 60, 90]), (15, list(range(0, 91, 15)))])
def test_window_boundaries_and_short_clips(stride, starts):
    rows = window_metadata([record(), record("short", 59), record("exact", 60)], {"stride": stride})
    a = [r for r in rows if r["clip_id"] == "a"]
    assert [r["start_frame"] for r in a] == starts
    assert all(r["end_frame"] == r["start_frame"] + 59 for r in rows)
    assert all(r["clip_id"] != "short" for r in rows)
    assert len([r for r in rows if r["clip_id"] == "exact"]) == 1
    assert len({r["window_id"] for r in rows}) == len(rows)


def test_window_uses_original_consecutive_frames(monkeypatch):
    full = torch.arange(150 * 50).reshape(150, 25, 2).float()
    def fake_item(self, index):
        return {"skeleton": full, "clip_id": "a", "path": "a", "label": 0, "activity": "action"}
    monkeypatch.setattr("src.data.dataset.SkeletonClipDataset.__getitem__", fake_item)
    dataset = SkeletonWindowDataset([record()], sequence_config={"stride": 30})
    assert torch.equal(dataset[1]["skeleton"], full[30:90])
    assert dataset[1]["window_id"] == "a__start0030_end0089"
    assert dataset[1]["num_source_frames"] == 150


def test_mean_logits_not_majority_vote_or_mean_probabilities():
    meta = [{"clip_id": "a", "path": "a", "subject_id": "s", "session_id": "D1"}] * 3
    # Majority vote AND mean probabilities pick class 0; mean logits picks 1.
    truth, pred, rows = aggregate_clip_logits([[2, 0], [2, 0], [0, 9]], meta, [1, 1, 1])
    assert truth == [1] and pred == [1]
    assert rows[0]["num_windows"] == 3
    assert np.isclose(rows[0]["mean_logit_1"], 3)
    with pytest.raises(ValueError, match="Inconsistent"):
        aggregate_clip_logits([[2, 0], [0, 2]], meta[:2], [0, 1])


def test_epoch_loss_diagnostics_account_for_short_last_batch():
    from src.engine import run_epoch
    from src.data.collate import skeleton_collate
    from torch.utils.data import DataLoader
    class Fixed(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x[:, 0, :]
    samples = [{"skeleton": torch.tensor([[v, 0.0]]), "label": 0, "clip_id": str(i), "path": str(i)}
               for i, v in enumerate([0.0, 0.0, 4.0])]
    stats = run_epoch(Fixed(), DataLoader(samples, batch_size=2, collate_fn=skeleton_collate),
                      torch.nn.CrossEntropyLoss(), torch.device("cpu"), ["a", "b"])
    a, b = np.log(2), np.log1p(np.exp(-4))
    assert stats["num_batches_per_epoch"] == 2
    assert np.isclose(stats["epoch_loss_mean"], (2 * a + b) / 3)
    assert np.isclose(stats["epoch_batch_loss_sum"], a + b)
