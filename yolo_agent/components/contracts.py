"""Unified component contracts and execution gates.

Component cards remain the backwards-compatible metadata format. Contracts
add the explicit tensor, deployment, compatibility, and maturity boundary
needed before a component can become an executable experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from yolo_agent.components.maturity import (
    MaturityName,
    maturity_rank,
)
from yolo_agent.components.schema import ComponentCard
from yolo_agent.core.yaml_io import YAMLModelMixin


ComponentValue = bool | Literal["unknown"]


class ComponentExecutionError(ValueError):
    """Raised when a component cannot be placed in an executable node."""


class ComponentContract(BaseModel, YAMLModelMixin):
    """The execution and compatibility contract for one component."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_contract.v1"
    component_id: str
    display_name: str
    category: str
    source_papers: list[str] = Field(default_factory=list)
    implementation_path: str | None = None
    adapter_class: str | None = None
    insertion_point: str = "unknown"
    supported_detector_families: list[str] = Field(default_factory=lambda: ["generic"])
    supported_model_patterns: list[str] = Field(default_factory=list)
    supported_heads: list[str] = Field(default_factory=lambda: ["generic"])
    incompatible_heads: list[str] = Field(default_factory=list)
    required_components: list[str] = Field(default_factory=list)
    conflicting_components: list[str] = Field(default_factory=list)
    replaces_components: list[str] = Field(default_factory=list)
    tensor_input_contract: dict[str, Any] = Field(default_factory=dict)
    tensor_output_contract: dict[str, Any] = Field(default_factory=dict)
    checkpoint_compatibility: str = "unknown"
    training_only: ComponentValue = "unknown"
    inference_only: ComponentValue = "unknown"
    changes_model_graph: ComponentValue = "unknown"
    affects_latency: str = "unknown"
    affects_model_size: str = "unknown"
    supports_amp: ComponentValue = "unknown"
    supports_ddp: ComponentValue = "unknown"
    supports_onnx: ComponentValue = "unknown"
    supports_tensorrt: ComponentValue = "unknown"
    fixed_imgsz_compatible: ComponentValue = "unknown"
    maturity: MaturityName = "metadata_only"
    tests_required: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)

    @field_validator("component_id", "display_name", "category")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("component_id, display_name, and category must not be empty")
        return value

    @property
    def maturity_rank(self) -> int:
        return maturity_rank(self.maturity)

    @property
    def can_execute(self) -> bool:
        """Whether this contract has passed the minimum execution gate."""
        return self.maturity_rank >= maturity_rank("smoke_passed")

    def assert_executable(
        self,
        *,
        detector_family: str | None = None,
        head: str | None = None,
        imgsz: int | None = None,
    ) -> None:
        """Raise a descriptive error unless the contract is executable."""
        if not self.can_execute:
            raise ComponentExecutionError(
                f"Component {self.component_id} is {self.maturity}; "
                "metadata/reference components cannot generate executable CommandSpec"
            )
        if not self.implementation_path or not self.adapter_class:
            raise ComponentExecutionError(
                f"Component {self.component_id} is smoke_passed but has no implementation_path/adapter_class"
            )
        if detector_family and self.supported_detector_families != ["generic"] and detector_family not in self.supported_detector_families:
            raise ComponentExecutionError(
                f"Component {self.component_id} does not support detector family {detector_family}"
            )
        if head and head in self.incompatible_heads:
            raise ComponentExecutionError(f"Component {self.component_id} is incompatible with head {head}")
        if imgsz is not None and self.fixed_imgsz_compatible is False and imgsz == 640:
            raise ComponentExecutionError(f"Component {self.component_id} is not compatible with fixed imgsz=640")


def contract_from_card(card: ComponentCard) -> ComponentContract:
    """Convert an old metadata card without granting execution maturity."""
    constraints = card.constraints
    maturity = (
        "reference_code_available"
        if any(key in constraints for key in ("requires_loss_patch", "requires_assigner_patch", "requires_architecture_patch"))
        else "metadata_only"
    )
    return ComponentContract(
        component_id=card.id,
        display_name=card.name,
        category=card.type,
        implementation_path=None,
        adapter_class=None,
        insertion_point=str(constraints.get("insertion_point", "unknown")),
        supported_detector_families=list(card.compatible_model_families),
        supported_model_patterns=list(card.compatible_model_families),
        supported_heads=[str(item) for item in constraints.get("supported_heads", ["generic"])],
        incompatible_heads=[str(item) for item in constraints.get("incompatible_heads", [])],
        required_components=[str(item) for item in constraints.get("requires", [])],
        conflicting_components=[str(item) for item in constraints.get("excludes", [])],
        training_only=constraints.get("training_only", "unknown"),
        inference_only=constraints.get("inference_only", "unknown"),
        changes_model_graph=constraints.get("changes_model_graph", "unknown"),
        supports_amp=constraints.get("supports_amp", "unknown"),
        supports_ddp=constraints.get("supports_ddp", "unknown"),
        supports_onnx=constraints.get("export_safe", "unknown"),
        supports_tensorrt=constraints.get("export_safe", "unknown"),
        fixed_imgsz_compatible=constraints.get("fixed_imgsz_compatible", "unknown"),
        maturity=maturity,
        tensor_input_contract={"compatibility_constraints": dict(constraints)},
        known_risks=list(card.risks),
        tests_required=list(constraints.get("tests_required", [])),
    )


def load_contracts(path: Path | str) -> list[ComponentContract]:
    """Load contracts from a mapping file or a directory of YAML files."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Component contract path does not exist: {source}")
    paths = [source] if source.is_file() else sorted(source.glob("*.yaml"))
    contracts: list[ComponentContract] = []
    for contract_path in paths:
        with contract_path.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        entries = raw.get("components", raw) if isinstance(raw, dict) else {}
        if not isinstance(entries, dict):
            raise ValueError(f"Component contract YAML must contain a mapping: {contract_path}")
        for component_id, values in entries.items():
            if not isinstance(values, dict):
                raise ValueError(f"Contract {component_id} must be a mapping")
            contracts.append(ComponentContract.model_validate({"component_id": component_id, **values}))
    return contracts


__all__ = [
    "ComponentContract",
    "ComponentExecutionError",
    "contract_from_card",
    "load_contracts",
]
