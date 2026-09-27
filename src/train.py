"""Shared training/configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping at top level of config: {config_path}")
    return config


def project_root_from_config_path(path: str | Path) -> Path:
    """Resolve repo root for the conventional configs/default.yaml layout."""

    config_path = Path(path).resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd().resolve()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
