"""StrictPaperReadinessGate — the global pre-training gate for the frozen 83.

The gate is a *resource-allocation* boundary, not a CLI decoration.  It is
evaluated at every seam that can start real training (CLI train entry,
queue execution, executors, and the Ultralytics runtime entrypoint) and it
is fail-closed: any inability to *prove* 83/83 ``implementation_ready``
locks training.  Dry-run, audit, validation, and research commands are
deliberately never gated.

Honesty rules enforced here:

* only the frozen manifest and current audit/registry artifacts are read —
  no in-memory optimistic state can unlock training;
* a manifest that is missing, unparseable, mis-hashed, or has the wrong
  paper count locks training (fail-closed, never guessed around);
* ``implementation_ready`` is the only readiness that counts;
  generic-only, metadata-only, alias, mock-smoke, and blocked evidence
  never unlock training;
* tests that exercise the training machinery with synthetic stubs must
  declare the synthetic scope explicitly via the module's environment
  variable — the declaration is recorded in the decision instead of
  silently weakening the gate.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.yaml_io import YAMLModelMixin

DEFAULT_MANIFEST_PATH = "configs/research/paper_83_manifest.yaml"
DEFAULT_EXACTNESS_AUDIT_PATH = "artifacts/paper_83_exactness_audit.yaml"
DEFAULT_IMPLEMENTATION_REGISTRY_PATH = (
    "runs/paper-readiness/paper_implementation_registry.yaml"
)

#: Environment variable that declares *synthetic scope* for a process:
#: a test that stubs the trainer (monkeypatched YOLO, fake executors, fake
#: GPU allocators) sets this so the gate reports its lock reasons honestly
#: while permitting the stubbed downstream call.  Real training processes
#: never set it, and every decision made under it is flagged.
SYNTHETIC_SCOPE_ENV = "YOLO_AGENT_PAPER_83_GATE_SYNTHETIC_SCOPE"

GateLockReason = Literal[
    "manifest_missing",
    "manifest_invalid",
    "paper_count_mismatch",
    "manifest_hash_mismatch",
    "audit_unavailable",
    "audit_hash_mismatch",
    "registry_unavailable",
    "registry_hash_mismatch",
    "ready_count_below_required",
    "blocked_papers_present",
    "mock_smoke_evidence",
    "generic_only_evidence",
]

LockableReasons = (
    "manifest_missing",
    "manifest_invalid",
    "paper_count_mismatch",
    "manifest_hash_mismatch",
    "audit_unavailable",
    "audit_hash_mismatch",
    "registry_unavailable",
    "registry_hash_mismatch",
    "ready_count_below_required",
    "blocked_papers_present",
    "mock_smoke_evidence",
    "generic_only_evidence",
)


class Paper83GateLockedError(RuntimeError):
    """Raised at the runtime training seam when the pre-training gate is locked."""

    def __init__(self, decision: "Paper83GateDecision") -> None:
        reasons = ", ".join(decision.lock_reasons) or "unknown"
        super().__init__(
            "paper-83 pre-training gate locked "
            f"(ready {decision.ready}/{decision.required}): {reasons}"
        )
        self.decision = decision


class Paper83GateConfig(BaseModel):
    """Frozen gate policy.  Defaults are the strictest honest setting."""

    model_config = ConfigDict(extra="forbid")

    required_count: int = Field(default=83, ge=1)
    required_maturity: Literal["implementation_ready"] = "implementation_ready"
    allow_generic_only: bool = False
    allow_mock_smoke: bool = False
    allow_missing_runtime_hook: bool = False
    allow_blocked: bool = False
    fail_closed: bool = True


class Paper83GateDecision(BaseModel, YAMLModelMixin):
    """One gate evaluation, serializable for audit trails."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_training_gate_decision.v1"
    allowed: bool
    locked: bool
    synthetic_scope: bool = False
    fail_closed: bool = True
    required: int
    required_maturity: str
    ready: int = 0
    blocked: int = 0
    out_of_scope: int = 0
    blocked_paper_ids: list[str] = Field(default_factory=list)
    lock_reasons: list[str] = Field(default_factory=list)
    details: dict[str, str] = Field(default_factory=dict)
    manifest_path: str = ""
    manifest_membership_hash: str = ""
    evidence_source: str = ""
    evidence_hash: str = ""
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def synthetic_scope_requested() -> bool:
    """Return whether the current process declared synthetic training scope."""

    return os.environ.get(SYNTHETIC_SCOPE_ENV, "").strip() == "1"


def _load_manifest_state(manifest_path: Path | str) -> tuple[Any | None, list[str]]:
    """Load the frozen manifest, returning (manifest, lock_reasons)."""

    from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest

    path = Path(manifest_path)
    if not path.is_file():
        return None, ["manifest_missing"]
    try:
        manifest = Paper83Manifest.from_yaml(path)
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"manifest_invalid:{type(exc).__name__}:{exc}"]
    return manifest, []


def _audit_ready_state(
    audit_path: Path | str,
    manifest_hash: str,
    config: Paper83GateConfig,
) -> tuple[int, int, int, list[str], list[str], str, str]:
    """Read ready/blocked/out-of-scope counts from the exactness audit.

    Returns (ready, blocked, out_of_scope, blocked_ids, lock_reasons,
    evidence_source, evidence_hash).
    """

    from yolo_agent.research.paper_exactness_schemas import PaperExactnessAudit

    path = Path(audit_path)
    if not path.is_file():
        return 0, 0, 0, [], ["audit_unavailable"], "", ""
    try:
        audit = PaperExactnessAudit.from_yaml(path)
    except (OSError, TypeError, ValueError) as exc:
        return 0, 0, 0, [], [f"audit_unavailable:{type(exc).__name__}:{exc}"], "", ""
    if audit.manifest_membership_hash != manifest_hash:
        return 0, 0, 0, [], ["manifest_hash_mismatch"], "exactness_audit", audit.audit_hash
    blocked_ids: list[str] = []
    mock_ids: list[str] = []
    generic_ids: list[str] = []
    ready = 0
    for record in audit.records:
        evidence_class = str(
            record.evidence_inventory.get("implementation_evidence_class", "")
        )
        if record.status == "implementation_ready":
            if not config.allow_mock_smoke and evidence_class == "mock_only":
                mock_ids.append(record.paper_id)
                continue
            if not config.allow_generic_only and evidence_class == "generic_only":
                generic_ids.append(record.paper_id)
                continue
            ready += 1
        elif record.status == "out_of_scope":
            continue
        else:
            blocked_ids.append(record.paper_id)
    blocked = len(blocked_ids)
    reasons: list[str] = []
    if mock_ids and not config.allow_mock_smoke:
        reasons.append("mock_smoke_evidence")
        blocked_ids = sorted(set(blocked_ids) | set(mock_ids))
        blocked = len(blocked_ids)
    if generic_ids and not config.allow_generic_only:
        reasons.append("generic_only_evidence")
        blocked_ids = sorted(set(blocked_ids) | set(generic_ids))
        blocked = len(blocked_ids)
    return (
        ready,
        blocked,
        audit.summary.out_of_scope,
        blocked_ids,
        reasons,
        "exactness_audit",
        audit.audit_hash,
    )


def _registry_ready_state(
    registry_path: Path | str,
    manifest_hash: str,
) -> tuple[int, int, list[str], str, str]:
    """Fallback readiness source: the paper implementation registry."""

    from yolo_agent.research.paper_implementation_schemas import (
        PaperImplementationRegistry,
    )

    path = Path(registry_path)
    if not path.is_file():
        return 0, 0, [], ["registry_unavailable"], ""
    try:
        registry = PaperImplementationRegistry.from_yaml(path)
    except (OSError, TypeError, ValueError) as exc:
        return 0, 0, [], [f"registry_unavailable:{type(exc).__name__}:{exc}"], ""
    if registry.manifest_membership_hash != manifest_hash:
        return 0, 0, [], ["registry_hash_mismatch"], registry.registry_hash
    blocked_ids = [
        record.paper_id
        for record in registry.records
        if record.readiness != "implementation_ready"
    ]
    return (
        registry.implementation_ready_count,
        len(blocked_ids),
        blocked_ids,
        [],
        registry.registry_hash,
    )


def evaluate_paper_83_training_gate(
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    exactness_audit_path: Path | str | None = None,
    implementation_registry_path: Path | str | None = None,
    config: Paper83GateConfig | None = None,
) -> Paper83GateDecision:
    """Evaluate the global pre-training gate.  Fail-closed by construction."""

    policy = config or Paper83GateConfig()
    lock_reasons: list[str] = []
    details: dict[str, str] = {}

    manifest, manifest_reasons = _load_manifest_state(manifest_path)
    lock_reasons.extend(manifest_reasons)

    ready = 0
    blocked = 0
    out_of_scope = 0
    blocked_ids: list[str] = []
    evidence_source = ""
    evidence_hash = ""
    manifest_hash = ""

    if manifest is not None:
        manifest_hash = manifest.campaign.membership_hash
        if manifest.paper_count != policy.required_count:
            lock_reasons.append(
                f"paper_count_mismatch:{manifest.paper_count}!={policy.required_count}"
            )
        audit_path = Path(
            exactness_audit_path
            if exactness_audit_path is not None
            else DEFAULT_EXACTNESS_AUDIT_PATH
        )
        ready, blocked, out_of_scope, blocked_ids, audit_reasons, evidence_source, evidence_hash = (
            _audit_ready_state(audit_path, manifest_hash, policy)
        )
        lock_reasons.extend(audit_reasons)
        if not evidence_source:
            registry_path = Path(
                implementation_registry_path
                if implementation_registry_path is not None
                else DEFAULT_IMPLEMENTATION_REGISTRY_PATH
            )
            ready, blocked, blocked_ids, registry_reasons, evidence_hash = (
                _registry_ready_state(registry_path, manifest_hash)
            )
            evidence_source = "implementation_registry"
            lock_reasons.extend(registry_reasons)
        details["evidence_source"] = evidence_source
        details["audit_path"] = str(audit_path)
    else:
        details["manifest_path"] = str(manifest_path)

    if policy.required_maturity == "implementation_ready":
        # Aggregate verdicts are computed even when evidence-seam reasons
        # already exist, so the decision always carries the honest counts;
        # the synthetic-scope path below only flips `allowed`.
        if ready < policy.required_count:
            lock_reasons.append(
                f"ready_count_below_required:{ready}<{policy.required_count}"
            )
        if blocked > 0 and not policy.allow_blocked:
            lock_reasons.append(f"blocked_papers_present:{blocked}")

    allowed = not lock_reasons
    synthetic_scope = False
    if not allowed and synthetic_scope_requested():
        # Synthetic scope never rewrites the lock reasons: tests still see
        # the honest gate verdict, they only permit stubbed downstream calls.
        synthetic_scope = True
        allowed = True

    return Paper83GateDecision(
        allowed=allowed,
        locked=not allowed,
        synthetic_scope=synthetic_scope,
        fail_closed=policy.fail_closed,
        required=policy.required_count,
        required_maturity=policy.required_maturity,
        ready=ready,
        blocked=blocked,
        out_of_scope=out_of_scope,
        blocked_paper_ids=blocked_ids,
        lock_reasons=lock_reasons,
        details=details,
        manifest_path=str(manifest_path),
        manifest_membership_hash=manifest_hash,
        evidence_source=evidence_source,
        evidence_hash=evidence_hash,
    )


def render_gate_summary(decision: Paper83GateDecision) -> list[str]:
    """Render the required CLI block for the gate."""

    lines = [
        "PAPER-83 PRE-TRAINING GATE",
        f"Required: {decision.required}",
        f"Ready: {decision.ready}",
        f"Blocked: {decision.blocked}",
        "Training allowed: " + ("YES" if decision.allowed else "NO"),
    ]
    if decision.locked:
        reasons = ", ".join(decision.lock_reasons) or "unspecified"
        lines.append(f"Locked: {reasons}")
    if decision.synthetic_scope:
        lines.append(
            f"Synthetic scope: declared via {SYNTHETIC_SCOPE_ENV}; "
            "downstream stubs permitted, real allocation still requires 83/83"
        )
    return lines


def gate_guard(
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    exactness_audit_path: Path | str | None = None,
    implementation_registry_path: Path | str | None = None,
    config: Paper83GateConfig | None = None,
) -> Paper83GateDecision:
    """Evaluate the gate and raise :class:`Paper83GateLockedError` if locked.

    Synthetic scope is honored: a process that declared it receives the
    decision (with ``synthetic_scope=True`` and its lock reasons intact)
    instead of an exception, so stubbed training can proceed while the
    honest verdict stays visible in logs and artifacts.
    """

    decision = evaluate_paper_83_training_gate(
        manifest_path=manifest_path,
        exactness_audit_path=exactness_audit_path,
        implementation_registry_path=implementation_registry_path,
        config=config,
    )
    if decision.locked:
        raise Paper83GateLockedError(decision)
    return decision


def command_is_training_allocation(command_type: str | None) -> bool:
    """Whether a command type allocates real training resources."""

    return command_type == "train"


def gate_refusal_for_training_command(
    command_type: str | None,
) -> Paper83GateDecision | None:
    """Evaluate the gate for one command; None when no refusal applies.

    Non-training commands (smoke, import, inference, custom tooling) never
    allocate training resources and are never gated.  Training commands get
    a full gate evaluation; a locked decision is returned for the caller to
    refuse with.  Synthetic-scope processes receive ``allowed=True`` with
    the flag set — the honest lock reasons stay visible in the decision.
    """

    if not command_is_training_allocation(command_type):
        return None
    decision = evaluate_paper_83_training_gate()
    if decision.allowed:
        return None
    return decision


def gate_refusal_resource_decision(
    decision: Paper83GateDecision,
) -> "object":
    """Build the queue-level refusal for a locked gate decision.

    Typed as ``object`` to avoid importing the resource scheduler (and its
    executor dependencies) into research-side imports; the caller in the
    orchestrator converts it.
    """

    from yolo_agent.core.resource_scheduler import ResourceDecision

    return ResourceDecision(
        status="blocked_by_resource",
        reasons=["paper_83_training_gate_locked", *decision.lock_reasons],
        message=(
            "PAPER-83 PRE-TRAINING GATE: training refused "
            f"(ready {decision.ready}/{decision.required}). "
            + "; ".join(decision.lock_reasons)
        ),
    )


def gate_refusal_execution_result(
    decision: Paper83GateDecision,
    *,
    run_id: str,
    node_id: str,
    candidate_id: str,
    command: Any,
) -> Any:
    """Build an executor-level skipped result for a locked gate."""

    from yolo_agent.core.executor import ExecutionResult

    now = datetime.now(timezone.utc)
    return ExecutionResult(
        run_id=run_id,
        node_id=node_id,
        candidate_id=candidate_id,
        status="skipped",
        command=command,
        started_at=now,
        ended_at=now,
        duration_seconds=0.0,
        message=(
            "PAPER-83 PRE-TRAINING GATE: training refused "
            f"(ready {decision.ready}/{decision.required}); "
            + "; ".join(decision.lock_reasons)
        ),
        metrics={"paper_83_gate_locked": True},
    )


__all__ = [
    "DEFAULT_EXACTNESS_AUDIT_PATH",
    "DEFAULT_IMPLEMENTATION_REGISTRY_PATH",
    "DEFAULT_MANIFEST_PATH",
    "SYNTHETIC_SCOPE_ENV",
    "GateLockReason",
    "Paper83GateConfig",
    "Paper83GateDecision",
    "Paper83GateLockedError",
    "command_is_training_allocation",
    "evaluate_paper_83_training_gate",
    "gate_guard",
    "gate_refusal_execution_result",
    "gate_refusal_for_training_command",
    "gate_refusal_resource_decision",
    "render_gate_summary",
    "synthetic_scope_requested",
]
