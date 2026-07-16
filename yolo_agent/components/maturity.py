"""Component implementation maturity and guarded state transitions."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from yolo_agent.components.contracts import ComponentContract
    from yolo_agent.core.event_log import EventLog


class ComponentMaturity(IntEnum):
    """Ordered maturity levels for a component implementation."""

    METADATA_ONLY = 0
    REFERENCE_CODE_AVAILABLE = 1
    ADAPTER_IMPLEMENTED = 2
    UNIT_TESTED = 3
    SMOKE_PASSED = 4
    PILOT_REPRODUCED = 5
    FULL_REPRODUCED = 6
    PRODUCTION_ELIGIBLE = 7


MaturityName = Literal[
    "metadata_only",
    "reference_code_available",
    "adapter_implemented",
    "unit_tested",
    "smoke_passed",
    "pilot_reproduced",
    "full_reproduced",
    "production_eligible",
]

_NAMES: tuple[MaturityName, ...] = (
    "metadata_only",
    "reference_code_available",
    "adapter_implemented",
    "unit_tested",
    "smoke_passed",
    "pilot_reproduced",
    "full_reproduced",
    "production_eligible",
)


class MaturityTransitionError(ValueError):
    """Raised when a component maturity transition is not allowed."""


class MaturityTransition(BaseModel):
    """Serializable description of a maturity transition."""

    component_id: str
    source: MaturityName
    target: MaturityName
    reason: str


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

    updated = contract.model_copy(update={"maturity": target})
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


__all__ = [
    "ComponentMaturity",
    "MaturityName",
    "MaturityTransition",
    "MaturityTransitionError",
    "can_transition",
    "maturity_rank",
    "transition_maturity",
]
