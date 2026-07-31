"""Component-specific certification gate before automatic queue admission."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.contracts import ComponentContract


class ComponentQueueCertificationResult(BaseModel):
    """Auditable component certification decision for one recipe."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    component_ids: list[str]
    report_path: Path | None = None
    report_hash: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    observed_capabilities: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)


class ComponentQueueCertificationGate:
    """Require effective smoke evidence before ASHA owns initial pilot budget.

    End-to-end pilot certification is an outcome of this queue, not a
    prerequisite for creating its first matched ``pilot_3`` assignment.
    """

    def evaluate(
        self,
        *,
        component_ids: list[str],
        report_path: Path | str | None,
        component_contracts: Mapping[str, ComponentContract] | None = None,
    ) -> ComponentQueueCertificationResult:
        components = sorted(set(component_ids))
        contracts = component_contracts or {}
        checks: dict[str, bool] = {}
        blockers: list[str] = []
        for component_id in components:
            contract = contracts.get(component_id)
            check_id = f"{component_id}:effective_smoke_passed"
            checks[check_id] = bool(contract is not None and contract.can_execute)
            if contract is None:
                blockers.append(f"effective_maturity_contract_missing:{component_id}")
            elif not contract.can_execute:
                blockers.append(
                    f"effective_maturity_below_smoke_passed:{component_id}:"
                    f"{contract.maturity}"
                )
        return ComponentQueueCertificationResult(
            allowed=not blockers,
            component_ids=components,
            report_path=Path(report_path) if report_path is not None else None,
            required_capabilities=["artifact_backed_smoke_passed"],
            checks=checks,
            blockers=blockers,
        )


__all__ = [
    "ComponentQueueCertificationGate",
    "ComponentQueueCertificationResult",
]
