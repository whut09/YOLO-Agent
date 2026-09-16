"""Paper-83 exactness audit schemas.

The exactness audit is a join layer over the six side audits and the paper
implementation registry.  It inventories evidence per paper, never grants new
mechanism claims, and maps every failing check to a blocker category from the
fixed campaign vocabulary.  "covered", "mapped", and "certified_adapter" are
deliberately absent: they are component-level descriptions and can never be
implementation-ready synonyms.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


EXACTNESS_CHECK_COUNT = 17

ExactnessCheckId = Literal[
    "membership_valid",
    "method_profile_valid",
    "mechanism_evidence_available",
    "implementation_spec_complete",
    "not_generic_only",
    "not_alias_only",
    "not_metadata_only",
    "not_no_op",
    "paper_specific_composition",
    "runtime_hook_real",
    "runtime_fingerprint_present",
    "unit_tests_present",
    "non_mock_smoke_present",
    "compatibility_tests_present",
    "rollback_path_present",
    "shared_primitive_correctly_referenced",
    "core_mechanism_covered",
]

EXACTNESS_CHECK_IDS: tuple[str, ...] = (
    "membership_valid",
    "method_profile_valid",
    "mechanism_evidence_available",
    "implementation_spec_complete",
    "not_generic_only",
    "not_alias_only",
    "not_metadata_only",
    "not_no_op",
    "paper_specific_composition",
    "runtime_hook_real",
    "runtime_fingerprint_present",
    "unit_tests_present",
    "non_mock_smoke_present",
    "compatibility_tests_present",
    "rollback_path_present",
    "shared_primitive_correctly_referenced",
    "core_mechanism_covered",
)

ExactnessPaperStatus = Literal[
    "implementation_ready",
    "blocked_missing_evidence",
    "blocked_missing_code",
    "blocked_runtime",
    "blocked_test",
    "blocked_compatibility",
    "blocked_asset",
    "blocked_license",
]

# Statuses are ordered by escalation: a paper inherits the most severe
# blocker category among its failing checks.  Blocking statuses come first.
BLOCKER_CATEGORY_ORDER: tuple[str, ...] = (
    "blocked_missing_evidence",
    "blocked_missing_code",
    "blocked_runtime",
    "blocked_test",
    "blocked_compatibility",
    "blocked_asset",
    "blocked_license",
)

BLOCKING_STATUSES: frozenset[str] = frozenset(BLOCKER_CATEGORY_ORDER)

# Forbid component-level descriptions as implementation-ready synonyms.
_FORBIDDEN_READY_SYNONYMS: frozenset[str] = frozenset(
    {"covered", "mapped", "certified_adapter"}
)

IMPLEMENTATION_READY = "implementation_ready"
OUT_OF_SCOPE_STATUS = "out_of_scope"

_EXACTNESS_SCHEMA_VERSION = "paper_exactness_record.v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ExactnessCheckResult(BaseModel):
    """One of the 17 exactness checks for one paper."""

    model_config = ConfigDict(extra="forbid")

    check_id: ExactnessCheckId
    passed: bool
    source: str
    detail: str = ""

    @model_validator(mode="after")
    def validate_result(self) -> "ExactnessCheckResult":
        if not self.source.strip():
            raise ValueError("exactness check requires a source reference")
        return self


class ExactnessPaperRecord(BaseModel):
    """Per-paper exactness status over the fixed 17-check inventory."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = _EXACTNESS_SCHEMA_VERSION
    paper_id: str
    title: str
    year: int = Field(ge=1900, le=2100)
    primary_domain: str
    secondary_domains: list[str] = Field(default_factory=list)
    manifest_membership_hash: str = Field(pattern=_SHA256_PATTERN)
    in_scope_for_implementation: bool
    checks: dict[str, ExactnessCheckResult]
    evidence_inventory: dict[str, Any] = Field(default_factory=dict)
    shared_primitives: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    side_audit_fingerprints: dict[str, str] = Field(default_factory=dict)
    implementation_fingerprint: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    side_status: dict[str, str] = Field(default_factory=dict)
    side_blockers: dict[str, list[str]] = Field(default_factory=dict)
    status: ExactnessPaperStatus | Literal["out_of_scope"]
    blockers: list[str] = Field(default_factory=list)
    blocker_category_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> "ExactnessPaperRecord":
        if not self.paper_id.strip():
            raise ValueError("exactness record requires paper_id")
        missing = sorted(set(EXACTNESS_CHECK_IDS) - set(self.checks))
        if missing:
            raise ValueError(
                "exactness record must inventory all "
                f"{EXACTNESS_CHECK_COUNT} checks; missing: {', '.join(missing)}"
            )
        extra = sorted(set(self.checks) - set(EXACTNESS_CHECK_IDS))
        if extra:
            raise ValueError(
                "exactness record carries unknown checks: " + ", ".join(extra)
            )
        if self.status == IMPLEMENTATION_READY:
            failed = [name for name, item in self.checks.items() if not item.passed]
            if failed:
                raise ValueError(
                    "implementation_ready record has failing checks: "
                    + ", ".join(failed)
                )
            if self.blockers:
                raise ValueError("implementation_ready record must not carry blockers")
            if not self.implementation_fingerprint:
                raise ValueError(
                    "implementation_ready record requires a runtime fingerprint"
                )
        if self.status in BLOCKING_STATUSES:
            if not self.blockers:
                raise ValueError(
                    f"{self.status} record must list at least one blocker"
                )
            for blocker in self.blockers:
                if not any(
                    blocker.startswith(category)
                    for category in BLOCKER_CATEGORY_ORDER
                ):
                    raise ValueError(
                        "blockers must carry a blocker-category prefix, got: "
                        + blocker
                    )
        if self.status == OUT_OF_SCOPE_STATUS and self.blockers:
            raise ValueError("out-of-scope record must not carry blockers")
        return self

    @property
    def failed_checks(self) -> list[str]:
        return sorted(name for name, item in self.checks.items() if not item.passed)

    @property
    def passed_check_count(self) -> int:
        return sum(1 for item in self.checks.values() if item.passed)


class ExactnessAuditSummary(BaseModel):
    """Aggregate counts over the exactness audit."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    ready: int = Field(ge=0)
    blocked: int = Field(ge=0)
    out_of_scope: int = Field(ge=0)
    by_status: dict[str, int] = Field(default_factory=dict)
    by_check: dict[str, int] = Field(default_factory=dict)
    full_check_pass_count: int = Field(ge=0, default=0)

    @model_validator(mode="after")
    def validate_summary(self) -> "ExactnessAuditSummary":
        if self.by_status and sum(self.by_status.values()) != self.total:
            raise ValueError("summary by_status counts must sum to total")
        if self.ready + self.blocked + self.out_of_scope != self.total:
            raise ValueError("ready + blocked + out_of_scope must equal total")
        return self


class PaperExactnessAudit(BaseModel, YAMLModelMixin):
    """Complete per-paper exactness audit over the frozen 83."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_exactness_audit.v1"
    manifest_path: str
    manifest_membership_hash: str = Field(pattern=_SHA256_PATTERN)
    implementation_registry_hash: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    paper_count: int = Field(ge=0)
    records: list[ExactnessPaperRecord] = Field(default_factory=list)
    summary: ExactnessAuditSummary
    audit_hash: str = ""

    @model_validator(mode="after")
    def validate_audit(self) -> "PaperExactnessAudit":
        ids = [item.paper_id for item in self.records]
        if ids != sorted(set(ids)):
            raise ValueError("exactness audit records must be sorted and unique")
        if self.paper_count != len(ids):
            raise ValueError("paper_count must equal record count")
        if self.audit_hash and self.audit_hash != self.calculate_hash():
            raise ValueError("exactness audit hash mismatch")
        return self

    def calculate_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "manifest_membership_hash": self.manifest_membership_hash,
                "implementation_registry_hash": self.implementation_registry_hash,
                "records": [item.model_dump(mode="json") for item in self.records],
                "summary": self.summary.model_dump(mode="json"),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_hash(self) -> "PaperExactnessAudit":
        object.__setattr__(
            self, "__dict__", {**self.__dict__, "audit_hash": self.calculate_hash()}
        )
        return self


class ExactnessGapEntry(BaseModel, YAMLModelMixin):
    """One actionable gap: what is missing, how to fix it, and what it needs."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    blocking_reason: str
    missing_requirement: str
    recommended_fix: str
    estimated_scope: Literal["small", "medium", "large"]
    dependency: str

    @model_validator(mode="after")
    def validate_entry(self) -> "ExactnessGapEntry":
        if not self.blocking_reason.strip():
            raise ValueError("gap entry requires blocking_reason")
        if not self.missing_requirement.strip():
            raise ValueError("gap entry requires missing_requirement")
        if not self.recommended_fix.strip():
            raise ValueError("gap entry requires recommended_fix")
        if not self.dependency.strip():
            raise ValueError("gap entry requires dependency")
        return self


class PaperExactnessGapQueue(BaseModel, YAMLModelMixin):
    """Queue of every gap in the exactness audit, ordered by paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_83_exactness_gap_queue.v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exactness_audit_hash: str = ""
    paper_count: int = Field(ge=0)
    blocked_paper_count: int = Field(ge=0)
    gaps: list[ExactnessGapEntry] = Field(default_factory=list)
    gaps_by_category: dict[str, int] = Field(default_factory=dict)
    queue_hash: str = ""

    @model_validator(mode="after")
    def validate_queue(self) -> "PaperExactnessGapQueue":
        ids = [item.paper_id for item in self.gaps]
        if ids != sorted(ids):
            raise ValueError("gap queue entries must be sorted by paper_id")
        if self.blocked_paper_count > self.paper_count:
            raise ValueError("blocked_paper_count cannot exceed paper_count")
        return self

    def calculate_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "exactness_audit_hash": self.exactness_audit_hash,
                "gaps": [item.model_dump(mode="json") for item in self.gaps],
                "gaps_by_category": self.gaps_by_category,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_hash(self) -> "PaperExactnessGapQueue":
        object.__setattr__(
            self,
            "__dict__",
            {**self.__dict__, "queue_hash": self.calculate_hash()},
        )
        return self


def validate_ready_status_vocabulary(status: str) -> str | None:
    """Return an error when *status* uses a forbidden ready-synonym."""

    if status in _FORBIDDEN_READY_SYNONYMS:
        return (
            f"status '{status}' is a component-level description and may not "
            "be used as an implementation-ready synonym"
        )
    return None


__all__ = [
    "BLOCKER_CATEGORY_ORDER",
    "BLOCKING_STATUSES",
    "EXACTNESS_CHECK_COUNT",
    "EXACTNESS_CHECK_IDS",
    "IMPLEMENTATION_READY",
    "OUT_OF_SCOPE_STATUS",
    "ExactnessAuditSummary",
    "ExactnessCheckId",
    "ExactnessCheckResult",
    "ExactnessGapEntry",
    "ExactnessPaperRecord",
    "ExactnessPaperStatus",
    "PaperExactnessAudit",
    "PaperExactnessGapQueue",
    "validate_ready_status_vocabulary",
]
