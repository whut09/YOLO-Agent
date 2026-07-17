"""Scoped, persistent consent for trusted full-budget training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.optimization_objective import OptimizationObjective, OptimizationObjectiveStatus
from yolo_agent.core.yaml_io import YAMLModelMixin


ConsentState = Literal["active", "invalidated", "exhausted", "completed"]
FullRunStage = Literal[
    "baseline_full",
    "baseline_confirm",
    "baseline_acceptance",
    "candidate_full",
    "candidate_confirmation",
    "completed",
    "stopped",
]


class FullRunConsent(BaseModel, YAMLModelMixin):
    """One explicit authorization bound to an immutable experiment scope."""

    schema_version: str = "1.0"
    consent_id: str
    run_id: str
    objective_hash: str
    dataset_manifest_sha256: str
    baseline_protocol_hash: str
    authorized_max_gpu_hours: float = Field(gt=0.0)
    state: ConsentState = "active"
    granted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    invalidated_at: datetime | None = None
    invalidation_reason: str | None = None

    @classmethod
    def grant(
        cls,
        *,
        run_id: str,
        objective: OptimizationObjective,
        dataset_manifest_sha256: str,
    ) -> "FullRunConsent":
        payload = {
            "run_id": run_id,
            "objective_hash": objective.objective_hash,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "baseline_protocol_hash": objective.baseline_protocol_hash,
            "authorized_max_gpu_hours": objective.max_gpu_hours,
        }
        consent_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(consent_id=consent_id, **payload)


class FullRunConsentDecision(BaseModel):
    """Validation result for a persisted full-run authorization."""

    allowed: bool
    reason: str
    consent: FullRunConsent | None = None
    gpu_hours_used: float = 0.0
    gpu_hours_remaining: float = 0.0


class FullRunStageStatus(BaseModel, YAMLModelMixin):
    """Compact terminal-facing state for the trusted full-run sequence."""

    schema_version: str = "1.0"
    stage: FullRunStage
    seed: int | None = None
    seed_total: int = 3
    progress: str = "pending"
    gpu_hours_used: float = 0.0
    gpu_hours_authorized: float = 0.0
    stop_reason: str = "continue"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FullRunConsentStore:
    """Persist consent and concise status under one run artifact directory."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.consent_path = self.run_dir / "artifacts" / "full_run_consent.yaml"
        self.status_path = self.run_dir / "artifacts" / "full_run_status.yaml"

    def load(self) -> FullRunConsent | None:
        return FullRunConsent.from_yaml(self.consent_path) if self.consent_path.is_file() else None

    def save(self, consent: FullRunConsent) -> Path:
        return _atomic_yaml(consent, self.consent_path)

    def save_status(self, status: FullRunStageStatus) -> Path:
        return _atomic_yaml(status, self.status_path)


class FullRunConsentDriver:
    """Grant and validate consent without allowing protocol or budget drift."""

    def __init__(self, run_dir: Path | str) -> None:
        self.store = FullRunConsentStore(run_dir)

    def grant(
        self,
        *,
        run_id: str,
        objective: OptimizationObjective,
        dataset_manifest_sha256: str | None,
    ) -> FullRunConsent:
        if not dataset_manifest_sha256:
            raise ValueError("full-run consent requires a dataset manifest sha256")
        consent = FullRunConsent.grant(
            run_id=run_id,
            objective=objective,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        self.store.save(consent)
        return consent

    def validate(
        self,
        *,
        run_id: str,
        objective: OptimizationObjective,
        dataset_manifest_sha256: str | None,
        objective_status: OptimizationObjectiveStatus | None = None,
    ) -> FullRunConsentDecision:
        consent = self.store.load()
        if consent is None:
            return FullRunConsentDecision(allowed=False, reason="full_run_consent_missing")
        mismatch = _scope_mismatch(consent, run_id, objective, dataset_manifest_sha256)
        used = objective_status.gpu_hours_used if objective_status is not None else 0.0
        remaining = max(0.0, consent.authorized_max_gpu_hours - used)
        if mismatch:
            invalidated = consent.model_copy(
                update={
                    "state": "invalidated",
                    "invalidated_at": datetime.now(timezone.utc),
                    "invalidation_reason": mismatch,
                }
            )
            self.store.save(invalidated)
            return FullRunConsentDecision(
                allowed=False,
                reason=mismatch,
                consent=invalidated,
                gpu_hours_used=used,
                gpu_hours_remaining=remaining,
            )
        if used >= consent.authorized_max_gpu_hours:
            exhausted = consent.model_copy(
                update={"state": "exhausted", "invalidation_reason": "gpu_budget_exhausted"}
            )
            self.store.save(exhausted)
            return FullRunConsentDecision(
                allowed=False,
                reason="gpu_budget_exhausted",
                consent=exhausted,
                gpu_hours_used=used,
                gpu_hours_remaining=0.0,
            )
        if consent.state != "active":
            return FullRunConsentDecision(
                allowed=False,
                reason=f"full_run_consent_{consent.state}",
                consent=consent,
                gpu_hours_used=used,
                gpu_hours_remaining=remaining,
            )
        return FullRunConsentDecision(
            allowed=True,
            reason="full_run_consent_valid",
            consent=consent,
            gpu_hours_used=used,
            gpu_hours_remaining=remaining,
        )


def _scope_mismatch(
    consent: FullRunConsent,
    run_id: str,
    objective: OptimizationObjective,
    dataset_manifest_sha256: str | None,
) -> str | None:
    if consent.run_id != run_id:
        return "full_run_consent_run_mismatch"
    if consent.objective_hash != objective.objective_hash:
        return "full_run_consent_objective_changed"
    if consent.dataset_manifest_sha256 != dataset_manifest_sha256:
        return "full_run_consent_dataset_manifest_changed"
    if consent.baseline_protocol_hash != objective.baseline_protocol_hash:
        return "full_run_consent_protocol_changed"
    if consent.authorized_max_gpu_hours != objective.max_gpu_hours:
        return "full_run_consent_gpu_budget_changed"
    return None


def _atomic_yaml(model: YAMLModelMixin, path: Path) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    model.to_yaml(temporary, exclude_none=True, sort_keys=False)
    temporary.replace(path)
    return path


__all__ = [
    "FullRunConsent",
    "FullRunConsentDecision",
    "FullRunConsentDriver",
    "FullRunConsentStore",
    "FullRunStageStatus",
]
