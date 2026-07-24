"""Reusable YAML/JSON serialization helpers for Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


class YAMLModelMixin:
    """Shared YAML/JSON serialization for Pydantic-backed harness models."""

    def to_yaml(
        self,
        path: Path | str,
        *,
        exclude_none: bool = False,
        sort_keys: bool = False,
        encoding: str = "utf-8-sig",
    ) -> Path:
        """Serialize the model to a YAML file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding=encoding) as file:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_none=exclude_none),
                file,
                sort_keys=sort_keys,
            )
        return output_path

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        *,
        encoding: str = "utf-8-sig",
    ) -> YAMLModelMixin:
        """Deserialize the model from a YAML file."""
        input_path = Path(path)
        with input_path.open("r", encoding=encoding) as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML must contain a mapping: {input_path}")
        return cls.model_validate(data)

    def to_json(
        self,
        path: Path | str,
        *,
        exclude_none: bool = False,
        sort_keys: bool = True,
        encoding: str = "utf-8-sig",
    ) -> Path:
        """Serialize the model to a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                self.model_dump(mode="json", exclude_none=exclude_none),
                indent=2,
                sort_keys=sort_keys,
            ),
            encoding=encoding,
        )
        return output_path
