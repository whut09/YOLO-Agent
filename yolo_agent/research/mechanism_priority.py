"""Configured implementation priority for canonical paper mechanisms."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.resources import ResourcePaths


class MechanismPriorityFamily(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: str
    priority_rank: int = Field(ge=1)
    canonical_component_ids: list[str] = Field(min_length=1)


class MechanismPriorityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "research_priority.v1"
    mechanism_families: list[MechanismPriorityFamily] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_components(self) -> "MechanismPriorityConfig":
        owners: dict[str, str] = {}
        for family in self.mechanism_families:
            for component_id in family.canonical_component_ids:
                previous = owners.get(component_id)
                if previous is not None:
                    raise ValueError(
                        f"canonical mechanism {component_id!r} belongs to both "
                        f"{previous!r} and {family.family_id!r}"
                    )
                owners[component_id] = family.family_id
        return self

    @classmethod
    def from_yaml(
        cls,
        path: Path | str = ResourcePaths.RESEARCH_PRIORITY,
    ) -> "MechanismPriorityConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)

    def priority_for(self, component_id: str) -> MechanismPriorityFamily | None:
        return next(
            (
                family
                for family in self.mechanism_families
                if component_id in family.canonical_component_ids
            ),
            None,
        )


__all__ = ["MechanismPriorityConfig", "MechanismPriorityFamily"]
