"""Small utilities shared by command-line entry points."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Circular base_config reference: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    base_name = config.get("base_config")
    if not base_name:
        return config
    base_path = Path(str(base_name))
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    override = dict(config)
    override.pop("base_config", None)
    return _deep_merge(_read_yaml(base_path, stack + (path,)), override)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and annotate it with the baseline project location."""
    config_path = Path(path).resolve()
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)
    # Config paths are stored under <project>/configs. Resolving relative
    # paths from that project keeps the CLI independent of the caller's cwd.
    config["_project_root"] = str(config_path.parent.parent.resolve())
    return config


def cfg_path(config: dict[str, Any], *keys: str, default: str) -> Path:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            value = default
            break
        value = value[key]
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(config.get("_project_root", Path.cwd())) / path
    return path.resolve()


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def json_dump(data: Any, path: str | Path) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        info["pytorch"] = torch.__version__
        info["torch_version_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["pytorch"] = None
        info["cuda_available"] = False
        info["gpu_name"] = None
    return info


def assert_finite(*arrays: Any) -> None:
    """Raise a useful error if any NumPy/Torch value contains NaN or Inf."""
    import numpy as np

    for value in arrays:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError("NaN or Inf detected")
