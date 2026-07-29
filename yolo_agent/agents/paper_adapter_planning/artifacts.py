"""Atomic serialization for paper adapter implementation plans."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import yaml

from yolo_agent.agents.paper_adapter_planning.schemas import PaperAdapterImplementationPlan


def write_implementation_plan(
    plan: PaperAdapterImplementationPlan,
    path: Path | str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.model_dump(mode="json")
    suffix = target.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("implementation plan path must end in .yaml, .yml, or .json")
    handle, temporary_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            if suffix == ".json":
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
            else:
                yaml.safe_dump(payload, file, sort_keys=False)
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


__all__ = ["write_implementation_plan"]
