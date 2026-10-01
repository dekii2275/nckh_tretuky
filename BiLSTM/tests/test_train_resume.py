import torch
from pathlib import Path
from src.engine import build_model
from src.utils import load_config


def test_resume_checkpoint_structure_and_loading(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/debug.yaml")
    model = build_model(config, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simulate saving checkpoint
    checkpoint_path = tmp_path / "last.pt"
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": None,
        "config": config,
        "class_names": ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"],
        "epoch": 5,
        "bad_epochs": 2,
        "val_macro_f1": 0.75,
        "monitor": "val_accuracy",
        "monitor_value": 0.80,
        "history": [{"epoch": i, "val_accuracy": 0.7 + i * 0.02} for i in range(1, 6)],
    }
    torch.save(payload, checkpoint_path)

    # Resume and test restoring
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    new_model = build_model(config, 10)
    new_model.load_state_dict(loaded["model_state"])

    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_optimizer.load_state_dict(loaded["optimizer_state"])

    assert loaded["epoch"] == 5
    assert loaded["bad_epochs"] == 2
    assert len(loaded["history"]) == 5
    assert loaded["monitor_value"] == 0.80
