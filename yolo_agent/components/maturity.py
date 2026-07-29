"""Component implementation maturity and guarded state transitions."""

from __future__ import annotations

import hashlib
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from yolo_agent.components.contracts import ComponentContract
    from yolo_agent.core.event_log import EventLog


class ComponentMaturity(IntEnum):
    """Ordered maturity levels for a component implementation."""

    METADATA_ONLY = 0
    RECIPE_IDEA_ONLY = 1
    ADAPTER_IMPLEMENTED = 2
    RUNTIME_INTEGRATED = 3
    UNIT_TESTED = 4
    SMOKE_PASSED = 5
    GPU_CERTIFIED = 6
    PILOT_REPRODUCED = 7
    FULL_REPRODUCED = 8
    CONFIRMED_MULTI_SEED = 9


MaturityName = Literal[
    "metadata_only",
    "recipe_idea_only",
    "adapter_implemented",
    "runtime_integrated",
    "unit_tested",
    "smoke_passed",
    "gpu_certified",
    "pilot_reproduced",
    "full_reproduced",
    "confirmed_multi_seed",
]

_NAMES: tuple[MaturityName, ...] = (
    "metadata_only",
    "recipe_idea_only",
    "adapter_implemented",
    "runtime_integrated",
    "unit_tested",
    "smoke_passed",
    "gpu_certified",
    "pilot_reproduced",
    "full_reproduced",
    "confirmed_multi_seed",
)


class MaturityTransitionError(ValueError):
    """Raised when a component maturity transition is not allowed."""


class MaturityTransition(BaseModel):
    """Serializable description of a maturity transition."""

    component_id: str
    source: MaturityName
    target: MaturityName
    reason: str


MaturityArtifactType = Literal[
    "recipe_prior",
    "adapter_source",
    "runtime_payload",
    "unit_test_report",
    "smoke_report",
    "gpu_certification_report",
    "pilot_paired_result",
    "full_reproduction_report",
    "multi_seed_confirmation_report",
]
ArtifactStatus = Literal["passed", "failed"]

_ARTIFACT_TARGETS: dict[MaturityName, MaturityArtifactType] = {
    "recipe_idea_only": "recipe_prior",
    "adapter_implemented": "adapter_source",
    "runtime_integrated": "runtime_payload",
    "unit_tested": "unit_test_report",
    "smoke_passed": "smoke_report",
    "gpu_certified": "gpu_certification_report",
    "pilot_reproduced": "pilot_paired_result",
    "full_reproduced": "full_reproduction_report",
    "confirmed_multi_seed": "multi_seed_confirmation_report",
}


class ComponentMaturityArtifact(BaseModel):
    """One immutable artifact contract supporting a maturity observation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "component_maturity_artifact.v1"
    component_id: str
    target_maturity: MaturityName
    artifact_type: MaturityArtifactType
    artifact_path: Path
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ArtifactStatus
    producer: str
    mock: bool = False
    protocol_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_type(self) -> "ComponentMaturityArtifact":
        expected = _ARTIFACT_TARGETS.get(self.target_maturity)
        if expected is None or expected != self.artifact_type:
            raise ValueError(
                f"artifact type {self.artifact_type} cannot support {self.target_maturity}"
            )
        return self

    def verify(self, root: Path | str | None = None) -> None:
        path = self.artifact_path
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        if not path.is_file():
            raise MaturityTransitionError(f"maturity artifact is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.artifact_sha256:
            raise MaturityTransitionError(
                f"maturity artifact hash mismatch: expected {self.artifact_sha256}, got {actual}"
            )


def maturity_artifact(
    *,
    component_id: str,
    target_maturity: MaturityName,
    artifact_path: Path | str,
    status: ArtifactStatus,
    producer: str,
    mock: bool = False,
    protocol_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ComponentMaturityArtifact:
    """Build a hash-bound artifact contract from an existing local file."""
    path = Path(artifact_path)
    return ComponentMaturityArtifact(
        component_id=component_id,
        target_maturity=target_maturity,
        artifact_type=_ARTIFACT_TARGETS[target_maturity],
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=status,
        producer=producer,
        mock=mock,
        protocol_hash=protocol_hash,
        metadata=metadata or {},
    )


def maturity_rank(value: MaturityName | str) -> int:
    """Return the stable ordinal for a maturity name."""
    try:
        return _NAMES.index(value)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ValueError(f"Unknown component maturity: {value}") from exc


def can_transition(source: MaturityName | str, target: MaturityName | str) -> bool:
    """Return whether the default state machine permits the transition."""
    return maturity_rank(target) == maturity_rank(source) + 1


def transition_maturity(
    contract: "ComponentContract",
    target: MaturityName,
    *,
    reason: str,
    event_log: "EventLog | None" = None,
    run_id: str | None = None,
    force: bool = False,
    artifact: ComponentMaturityArtifact | None = None,
    artifact_root: Path | str | None = None,
) -> "ComponentContract":
    """Advance a contract by one level and record the change.

    Forward transitions cannot skip levels. Backward transitions are only
    allowed with ``force=True`` and a non-empty reason, so a failed
    implementation can be demoted without silently changing its history.
    """
    source = contract.maturity
    source_rank = maturity_rank(source)
    target_rank = maturity_rank(target)
    if target_rank == source_rank:
        return contract
    if target_rank != source_rank + 1 and not (force and target_rank < source_rank):
        raise MaturityTransitionError(
            f"Invalid maturity transition for {contract.component_id}: {source} -> {target}"
        )
    if not reason.strip():
        raise MaturityTransitionError("A maturity transition requires a reason")

    artifacts = list(contract.maturity_artifacts)
    if target_rank > source_rank:
        if artifact is None:
            raise MaturityTransitionError(
                f"{source} -> {target} requires a { _ARTIFACT_TARGETS[target] } artifact"
            )
        _validate_transition_artifact(
            contract.component_id,
            target,
            artifact,
            artifact_root=artifact_root,
        )
        artifacts.append(artifact)

    updated = contract.model_copy(update={"maturity": target, "maturity_artifacts": artifacts})
    if event_log is not None:
        event_log.append(
            run_id=run_id or "component-contract",
            event_type="component_maturity_changed",
            message=f"Component {contract.component_id} maturity: {source} -> {target}",
            details={
                "component_id": contract.component_id,
                "from": source,
                "to": target,
                "reason": reason,
                "forced": force and target_rank < source_rank,
            },
        )
    return updated


def record_maturity_artifact(
    contract: "ComponentContract",
    artifact: ComponentMaturityArtifact,
    *,
    artifact_root: Path | str | None = None,
) -> "ComponentContract":
    """Retain failed/mock evidence without promoting component maturity."""
    if artifact.component_id != contract.component_id:
        raise MaturityTransitionError("maturity artifact component_id mismatch")
    artifact.verify(artifact_root)
    return contract.model_copy(
        update={"maturity_artifacts": [*contract.maturity_artifacts, artifact]}
    )


def _validate_transition_artifact(
    component_id: str,
    target: MaturityName,
    artifact: ComponentMaturityArtifact,
    *,
    artifact_root: Path | str | None,
) -> None:
    if artifact.component_id != component_id:
        raise MaturityTransitionError("maturity artifact component_id mismatch")
    if artifact.target_maturity != target:
        raise MaturityTransitionError("maturity artifact target mismatch")
    if artifact.status != "passed":
        raise MaturityTransitionError("failed maturity artifact cannot promote status")
    if artifact.mock and maturity_rank(target) >= maturity_rank("smoke_passed"):
        raise MaturityTransitionError("mock evidence cannot promote smoke or higher maturity")
    artifact.verify(artifact_root)


__all__ = [
    "ComponentMaturity",
    "ComponentMaturityArtifact",
    "MaturityName",
    "MaturityTransition",
    "MaturityTransitionError",
    "can_transition",
    "maturity_rank",
    "maturity_artifact",
    "record_maturity_artifact",
    "transition_maturity",
]
