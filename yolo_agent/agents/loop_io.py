"""Small structured IO helpers for loop harness artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> Any:
    """Read a JSON artifact."""
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping artifact."""
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return data


def write_yaml(path: Path, data: Any) -> None:
    """Write a YAML artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
