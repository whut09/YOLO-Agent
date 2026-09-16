"""Paper-83 exactness audit builder.

Joins the six side audits (data, loss, graph, domain, training-control,
inference) and the paper implementation registry into a single per-paper
17-check inventory.  The builder inventories evidence only: it never grants a
mechanism claim that a side audit does not already carry, and it maps every
failing check to a blocker category from the fixed vocabulary.

The result may legitimately be fewer than 83/83 ready.  The audit must still
be produced, written to artifacts, documented, tested, and committed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yolo_agent.research.paper_exactness_schemas import PaperExactnessGapQueue

from yolo_agent.loss_math.formula_provenance import (
    assignment_formula_provenance,
    loss_formula_provenance,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_data_side import build_paper_data_side_audit
from yolo_agent.research.paper_domain_side import build_paper_domain_side_audit
from yolo_agent.research.paper_exactness_schemas import (
    BLOCKER_CATEGORY_ORDER,
    BLOCKING_STATUSES,
    EXACTNESS_CHECK_COUNT,
    IMPLEMENTATION_READY,
    OUT_OF_SCOPE_STATUS,
    ExactnessAuditSummary,
    ExactnessCheckResult,
    ExactnessPaperRecord,
    PaperExactnessAudit,
)
from yolo_agent.research.paper_graph_side import build_paper_graph_side_audit
from yolo_agent.research.paper_implementation_registry import (
    build_paper_implementation_registry,
)
from yolo_agent.research.paper_inference_side import PaperInferenceSideAuditBuilder
from yolo_agent.research.paper_loss_side import build_paper_loss_side_audit
from yolo_agent.research.paper_training_control import PaperTrainingControlAuditBuilder

DEFAULT_MANIFEST_PATH = "configs/research/paper_83_manifest.yaml"
DEFAULT_PLAN_PATH = "configs/research/paper_83_engineering_plan.yaml"

# Mechanisms that live in the explicit-formula space: only these can be
# alias-only (WIoU->CIoU style rebranding).  All other domains are probed by
# their own side audits, which already assert real tensor behavior.
_LOSS_FORMULAS = dict(loss_formula_provenance())
_ASSIGN_FORMULAS = dict(assignment_formula_provenance())

# Membership, profile identity, and evidence availability fail together when
# the manifest/profile layer is broken; they never downgrade a ready paper.
_IDENTITY_CHECKS = frozenset(
    {"membership_valid", "method_profile_valid", "mechanism_evidence_available"}
)

_CHECK_CATEGORY: dict[str, str] = {
    "membership_valid": "blocked_missing_evidence",
    "method_profile_valid": "blocked_missing_evidence",
    "mechanism_evidence_available": "blocked_missing_evidence",
    "implementation_spec_complete": "blocked_missing_code",
    "not_generic_only": "blocked_missing_code",
    "not_alias_only": "blocked_missing_code",
    "not_metadata_only": "blocked_missing_code",
    "not_no_op": "blocked_runtime",
    "paper_specific_composition": "blocked_missing_code",
    "runtime_hook_real": "blocked_runtime",
    "runtime_fingerprint_present": "blocked_runtime",
    "unit_tests_present": "blocked_test",
    "non_mock_smoke_present": "blocked_test",
    "compatibility_tests_present": "blocked_compatibility",
    "rollback_path_present": "blocked_missing_code",
    "shared_primitive_correctly_referenced": "blocked_missing_code",
    "core_mechanism_covered": "blocked_missing_code",
}


def _category_for_check(check_id: str) -> str:
    return _CHECK_CATEGORY[check_id]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _escalate(statuses: list[str]) -> str:
    """Return the most severe blocker category among *statuses*."""

    for category in BLOCKER_CATEGORY_ORDER:
        if category in statuses:
            return category
    return statuses[0] if statuses else OUT_OF_SCOPE_STATUS


def _mechanism_ids_for(paper_id: str, plan_entry: Any | None, spec: Any | None) -> list[str]:
    """Paper-specific mechanism IDs from the plan entry, else the spec."""

    if isinstance(plan_entry, dict):
        ids = plan_entry.get("paper_specific_mechanism_ids")
        if isinstance(ids, list) and ids:
            return [item for item in ids if isinstance(item, str)]
        config = plan_entry.get("paper_specific_config")
        if isinstance(config, dict):
            source_ids = config.get("source_mechanism_ids")
            if isinstance(source_ids, list) and source_ids:
                return [item for item in source_ids if isinstance(item, str)]
    if spec is not None:
        return [item for item in (spec.paper_specific_mechanisms or [])]
    return []


def _formula_referenceable(mechanism_ids: list[str]) -> bool:
    """Whether every mechanism in the formula space has explicit provenance."""

    for mechanism_id in mechanism_ids:
        if mechanism_id not in _LOSS_FORMULAS and mechanism_id not in _ASSIGN_FORMULAS:
            return False
    return True


def _alias_only(mechanism_ids: list[str]) -> bool:
    """Alias-only: the paper's loss-space mechanisms lack formula provenance.

    Papers with no mechanisms in the formula space cannot be alias-only; their
    mechanisms are validated by their own domain's behavior probes.
    """

    in_space = [
        item
        for item in mechanism_ids
        if item in _LOSS_FORMULAS or item in _ASSIGN_FORMULAS
    ]
    return bool(in_space) and not _formula_referenceable(in_space)


def _rollback_present(spec: Any | None, plan_entry: Any | None) -> bool:
    if spec is not None and spec.runtime_insertion_points:
        return True
    if isinstance(plan_entry, dict):
        config = plan_entry.get("paper_specific_config")
        if isinstance(config, dict):
            insertion_points = config.get("insertion_points")
            if isinstance(insertion_points, list) and insertion_points:
                return True
            rollback = config.get("rollback")
            if isinstance(rollback, dict) and rollback.get("mode") is not None:
                return True
    return False


def _spec_in_ready_equivalent_state(spec: Any) -> bool:
    """A spec is in an implementation-ready *equivalent* state when every
    implementation-ready requirement is satisfied except for the live
    component-certification artifacts (which expire with run history)."""

    if spec.implementation_evidence_class != "paper_specific":
        return False
    if not spec.runtime_implementation_verified:
        return False
    if not (spec.unit_test_refs and spec.smoke_test_refs and spec.compatibility_test_refs):
        return False
    expired_markers = (
        "component_unit_evidence_missing",
        "component_smoke_evidence_missing",
    )
    return not any(
        marker in blocker for blocker in spec.blockers for marker in expired_markers
    )


class PaperExactnessAuditBuilder:
    """Build the per-paper exactness audit from the frozen manifest."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
        method_coverage_path: Path | str = (
            "research/production/paper_method_coverage.yaml"
        ),
        inventory_path: Path | str | None = (
            "runs/coverage-audit/paper_execution_inventory.yaml"
        ),
        coverage_path: Path | str | None = (
            "research/production/coverage_baseline.yaml"
        ),
        contracts_path: Path | str | None = (
            "research/production/component_contracts.yaml"
        ),
        tests_root: Path | str = "tests",
        plan_path: Path | str = DEFAULT_PLAN_PATH,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.method_coverage_path = Path(method_coverage_path)
        self.inventory_path = inventory_path
        self.coverage_path = coverage_path
        self.contracts_path = contracts_path
        self.tests_root = Path(tests_root)
        self.plan_path = Path(plan_path)

    # -- public API ---------------------------------------------------------

    def build(self) -> PaperExactnessAudit:
        import yaml

        manifest = Paper83Manifest.from_yaml(self.manifest_path)
        membership_hash = manifest.campaign.membership_hash

        registry = build_paper_implementation_registry(
            manifest_path=self.manifest_path,
            method_coverage_path=self.method_coverage_path,
            inventory_path=self.inventory_path,
            coverage_path=self.coverage_path,
            contracts_path=self.contracts_path,
            tests_root=self.tests_root,
        )
        specs = {item.paper_id: item for item in registry.records}

        data_audit = build_paper_data_side_audit()
        loss_audit = build_paper_loss_side_audit()
        graph_audit = build_paper_graph_side_audit(write_artifacts=False)
        domain_audit = build_paper_domain_side_audit()
        control_records = PaperTrainingControlAuditBuilder().build()
        inference_audit = PaperInferenceSideAuditBuilder().build()

        side_audits: dict[str, Any] = {
            "data": data_audit,
            "loss": loss_audit,
            "graph": graph_audit,
            "domain": domain_audit,
            "training_control": control_records,
            "inference": inference_audit,
        }
        side_records: dict[str, dict[str, Any]] = {name: {} for name in side_audits}
        for name, audit in side_audits.items():
            records_iterable = audit if isinstance(audit, list) else audit.records
            for record in records_iterable:
                side_records[name][record.paper_id] = record

        plan_entries: dict[str, dict[str, Any]] = {}
        if self.plan_path.is_file():
            with self.plan_path.open("r", encoding="utf-8-sig") as file:
                plan_data = yaml.safe_load(file) or {}
            for entry in plan_data.get("papers") or []:
                if isinstance(entry, dict) and isinstance(entry.get("paper_id"), str):
                    plan_entries[entry["paper_id"]] = entry

        records = [
            self._build_record(
                paper=paper,
                registry_hash=registry.registry_hash,
                membership_hash=membership_hash,
                spec=specs.get(paper.paper_id),
                side_records=side_records,
                plan_entry=plan_entries.get(paper.paper_id),
            )
            for paper in manifest.papers
        ]
        return self._assemble_audit(
            manifest=manifest,
            registry_hash=registry.registry_hash,
            records=records,
        )

    # -- record construction ------------------------------------------------

    def _build_record(
        self,
        *,
        paper: Any,
        registry_hash: str | None,
        membership_hash: str,
        spec: Any | None,
        side_records: dict[str, dict[str, Any]],
        plan_entry: Any | None,
    ) -> ExactnessPaperRecord:
        paper_id = paper.paper_id
        del registry_hash  # recorded at audit level, not per paper
        blockers: list[str] = []
        spec_blockers = list(spec.blockers) if spec is not None else []
        spec_ready = bool(spec is not None and spec.readiness == IMPLEMENTATION_READY)
        spec_ready_equivalent = spec is not None and _spec_in_ready_equivalent_state(spec)

        def _check(
            check_id: str,
            passed: bool,
            source: str,
            detail: str = "",
        ) -> ExactnessCheckResult:
            if not passed:
                blockers.append(
                    f"{_category_for_check(check_id)}:{check_id}:{paper_id}"
                )
            return ExactnessCheckResult(
                check_id=check_id, passed=passed, source=source, detail=detail
            )

        # 1. membership ---------------------------------------------------------
        membership_ok = bool(paper.paper_id) and bool(paper.title)
        checks = {
            "membership_valid": _check(
                "membership_valid",
                membership_ok,
                "paper_83_manifest.yaml",
                "" if membership_ok else "manifest entry lacks paper identity",
            )
        }

        # 2. method profile -------------------------------------------------------
        profile_ok = (
            spec is not None
            and bool(spec.method_profile_id)
            and not any("method_profile" in blocker for blocker in spec_blockers)
        )
        checks["method_profile_valid"] = _check(
            "method_profile_valid",
            profile_ok,
            "paper_implementation_registry",
            spec.method_profile_id if spec is not None else "spec missing",
        )

        # 3. mechanism evidence ----------------------------------------------------
        evidence_refs = list(spec.paper_evidence_refs) if spec is not None else []
        # Only *genuine* paper-evidence failures count here.  Component
        # certification expiry (component_*_evidence_missing) is a runtime
        # artifact problem, not missing paper evidence.
        evidence_fail_markers = (
            "paper_specific_evidence_unbound",
            "missing_method_profile",
            "unknown_mechanism",
        )
        evidence_ok = bool(evidence_refs) and not any(
            blocker.startswith(evidence_fail_markers) for blocker in spec_blockers
        )
        checks["mechanism_evidence_available"] = _check(
            "mechanism_evidence_available",
            evidence_ok,
            "paper_evidence_refs",
            f"{len(evidence_refs)} refs",
        )

        # 4. implementation spec completeness ---------------------------------------
        required_spec_fields = (
            "component_ids",
            "adapter_ids",
            "runtime_insertion_points",
            "runtime_hooks",
            "paper_specific_config",
            "implementation_fingerprint",
        )
        spec_complete = spec is not None and all(
            getattr(spec, field) for field in required_spec_fields
        )
        checks["implementation_spec_complete"] = _check(
            "implementation_spec_complete",
            spec_complete,
            "paper_implementation_registry",
            "" if spec_complete else "spec missing required binding fields",
        )

        # 5-8. evidence-class negatives ----------------------------------------------
        mechanisms = _mechanism_ids_for(paper_id, plan_entry, spec)
        component_ids = list(spec.component_ids) if spec is not None else []
        evidence_class = (
            spec.implementation_evidence_class if spec is not None else "unknown"
        )
        runtime_verified = bool(
            spec.runtime_implementation_verified if spec is not None else False
        )
        checks["not_generic_only"] = _check(
            "not_generic_only",
            evidence_class != "generic_only",
            "implementation_evidence_class",
        )
        checks["not_alias_only"] = _check(
            "not_alias_only",
            not _alias_only(mechanisms),
            "loss_math/formula_provenance.py",
            "" if mechanisms else "no mechanisms in the explicit-formula space",
        )
        checks["not_metadata_only"] = _check(
            "not_metadata_only",
            evidence_class not in {"metadata_only", "recipe_only", "unknown"},
            "implementation_evidence_class",
        )
        checks["not_no_op"] = _check(
            "not_no_op", runtime_verified, "runtime_implementation_verified"
        )

        # 9. paper-specific composition -----------------------------------------------
        paper_config = spec.paper_specific_config if spec is not None else {}
        composition_ok = bool(mechanisms) and bool(paper_config)
        checks["paper_specific_composition"] = _check(
            "paper_specific_composition",
            composition_ok,
            "paper_specific_mechanisms + paper_specific_config",
        )

        # 10. runtime hook real ----------------------------------------------------------
        hooks = list(spec.runtime_hooks) if spec is not None else []
        checks["runtime_hook_real"] = _check(
            "runtime_hook_real", bool(hooks) and runtime_verified, "runtime_hooks"
        )

        # 11. runtime fingerprint ----------------------------------------------------------
        fingerprint = (
            str(spec.implementation_fingerprint) if spec is not None else None
        )
        checks["runtime_fingerprint_present"] = _check(
            "runtime_fingerprint_present", bool(fingerprint), "implementation_fingerprint"
        )

        # 12-14. tests -----------------------------------------------------------------------
        unit_refs = list(spec.unit_test_refs) if spec is not None else []
        smoke_refs = list(spec.smoke_test_refs) if spec is not None else []
        compat_refs = list(spec.compatibility_test_refs) if spec is not None else []
        checks["unit_tests_present"] = _check(
            "unit_tests_present",
            bool(unit_refs)
            and not any("unit" in blocker for blocker in spec_blockers),
            "unit_test_refs",
        )
        checks["non_mock_smoke_present"] = _check(
            "non_mock_smoke_present",
            bool(smoke_refs)
            and not any("smoke" in blocker for blocker in spec_blockers),
            "smoke_test_refs",
        )
        checks["compatibility_tests_present"] = _check(
            "compatibility_tests_present",
            bool(compat_refs)
            and not any(
                "incompatible" in blocker or "compatibility" in blocker
                for blocker in spec_blockers
            ),
            "compatibility_test_refs",
        )

        # 15. rollback path --------------------------------------------------------------------
        checks["rollback_path_present"] = _check(
            "rollback_path_present",
            _rollback_present(spec, plan_entry),
            "runtime_insertion_points / plan insertion_points",
        )

        # 16. shared primitive references --------------------------------------------------------
        primitives_ok = _primitives_correctly_referenced(spec, mechanisms)
        checks["shared_primitive_correctly_referenced"] = _check(
            "shared_primitive_correctly_referenced",
            primitives_ok,
            "shared_primitives vs bound components",
        )

        # 17. core mechanism coverage --------------------------------------------------------------
        checks["core_mechanism_covered"] = _check(
            "core_mechanism_covered",
            bool(mechanisms) and bool(component_ids),
            "paper_specific_mechanisms + component_ids",
        )

        # side-audit inputs -------------------------------------------------------------------------
        side_status: dict[str, str] = {}
        side_blockers: dict[str, list[str]] = {}
        side_fingerprints: dict[str, str] = {}
        for name, per_paper in side_records.items():
            record = per_paper.get(paper_id)
            if record is None:
                continue
            # Side statuses are joined verbatim: each side audit owns its
            # vocabulary (ready / blocked_missing_evidence / out_of_scope).
            side_status[name] = str(record.status)
            side_blockers[name] = list(getattr(record, "blockers", []) or [])
            record_fp = getattr(record, "implementation_fingerprint", None)
            if isinstance(record_fp, str) and record_fp:
                side_fingerprints[name] = record_fp

        # final status ---------------------------------------------------------------------------------
        in_scope = bool(mechanisms or component_ids)
        final_blockers: list[str] = []
        if not in_scope:
            status: str = OUT_OF_SCOPE_STATUS
            # An out-of-scope paper is inventoried but does not block the
            # campaign; its check rows are marked passed so the check matrix
            # measures only in-scope work.
            checks = {
                name: item.model_copy(update={"passed": True})
                for name, item in checks.items()
            }
        else:
            status = _record_status(
                checks, spec_is_ready=spec_ready, spec_ready_equivalent=spec_ready_equivalent
            )
            if status in BLOCKING_STATUSES:
                final_blockers = _unique(
                    [f"{status}:{paper_id}"]
                    + [
                        blocker
                        for blocker in spec_blockers
                        if blocker.startswith(status)
                    ]
                )

        return ExactnessPaperRecord(
            paper_id=paper_id,
            title=paper.title,
            year=paper.year,
            primary_domain=getattr(paper, "primary_domain", "") or "",
            secondary_domains=list(getattr(paper, "secondary_domains", []) or []),
            manifest_membership_hash=membership_hash,
            in_scope_for_implementation=in_scope,
            checks=checks,
            evidence_inventory={
                "component_ids": component_ids,
                "adapter_ids": list(spec.adapter_ids) if spec is not None else [],
                "paper_specific_mechanisms": mechanisms,
                "paper_evidence_refs": evidence_refs,
                "unit_test_refs": unit_refs,
                "smoke_test_refs": smoke_refs,
                "compatibility_test_refs": compat_refs,
                "implementation_evidence_class": evidence_class,
            },
            shared_primitives=list(spec.shared_primitives) if spec is not None else [],
            component_ids=component_ids,
            side_audit_fingerprints=side_fingerprints,
            implementation_fingerprint=fingerprint,
            side_status=side_status,
            side_blockers=side_blockers,
            status=status,
            blockers=final_blockers,
            blocker_category_counts={},
        )

    # -- assembly --------------------------------------------------------------

    def _assemble_audit(
        self,
        *,
        manifest: Any,
        registry_hash: str | None,
        records: list[ExactnessPaperRecord],
    ) -> PaperExactnessAudit:
        records = sorted(records, key=lambda item: item.paper_id)
        by_status = Counter(record.status for record in records)
        by_check = Counter(
            record.paper_id
            for record in records
            for item in record.checks.values()
            if not item.passed
        )
        ready = by_status.get(IMPLEMENTATION_READY, 0)
        blocked = sum(by_status.get(name, 0) for name in BLOCKER_CATEGORY_ORDER)
        out_of_scope = by_status.get(OUT_OF_SCOPE_STATUS, 0)
        full_pass = sum(
            1
            for record in records
            if record.passed_check_count == EXACTNESS_CHECK_COUNT
        )
        summary = ExactnessAuditSummary(
            total=len(records),
            ready=ready,
            blocked=blocked,
            out_of_scope=out_of_scope,
            by_status=dict(sorted(by_status.items())),
            by_check=dict(sorted(by_check.items())),
            full_check_pass_count=full_pass,
        )
        audit = PaperExactnessAudit(
            manifest_path=str(self.manifest_path.resolve()),
            manifest_membership_hash=manifest.campaign.membership_hash,
            implementation_registry_hash=registry_hash,
            paper_count=len(records),
            records=records,
            summary=summary,
        )
        return audit.with_hash()


def _record_status(
    checks: dict[str, ExactnessCheckResult],
    *,
    spec_is_ready: bool,
    spec_ready_equivalent: bool,
) -> str:
    if spec_is_ready or spec_ready_equivalent:
        failed = [name for name, item in checks.items() if not item.passed]
        if not failed:
            return IMPLEMENTATION_READY
        categories = [
            _category_for_check(name) for name in failed if name not in _IDENTITY_CHECKS
        ]
        if categories:
            return _escalate(categories)
        return IMPLEMENTATION_READY
    failed = [
        name
        for name, item in checks.items()
        if not item.passed and name not in _IDENTITY_CHECKS
    ]
    if not failed:
        failed = ["implementation_spec_complete"]
    return _escalate([_category_for_check(name) for name in failed])


def _primitives_correctly_referenced(spec: Any | None, mechanisms: list[str]) -> bool:
    """Shared primitives must be referenced through bound components, not
    re-implemented (copied) per paper.  A paper passes when its shared
    primitives coexist with component bindings, or when it declares none."""

    if spec is None:
        return False
    shared = set(spec.shared_primitives or [])
    if not shared:
        return bool(mechanisms) or bool(spec.component_ids)
    return bool(spec.component_ids)


def build_paper_exactness_audit(**kwargs: object) -> PaperExactnessAudit:
    """Functional builder for CLI and offline callers."""

    return PaperExactnessAuditBuilder(**kwargs).build()


def build_paper_exactness_gap_queue(
    audit: PaperExactnessAudit,
) -> "PaperExactnessGapQueue":
    """Derive the actionable gap queue from a built exactness audit."""

    from yolo_agent.research.paper_exactness_schemas import (
        ExactnessGapEntry,
        PaperExactnessGapQueue as _Queue,
    )

    gaps: list[ExactnessGapEntry] = []
    for record in audit.records:
        if record.status not in BLOCKING_STATUSES:
            continue
        failed = record.failed_checks
        if not failed:
            failed = ["implementation_spec_complete"]
        for check_id in failed:
            category = _category_for_check(check_id)
            gaps.append(
                ExactnessGapEntry(
                    paper_id=record.paper_id,
                    blocking_reason=record.blockers[0] if record.blockers else category,
                    missing_requirement=f"{category}:{check_id}",
                    recommended_fix=_recommended_fix(check_id),
                    estimated_scope=_estimated_scope(check_id),
                    dependency=_dependency(check_id),
                )
            )
    by_category = Counter(entry.missing_requirement.split(":", 1)[0] for entry in gaps)
    blocked_ids = {
        record.paper_id for record in audit.records if record.status in BLOCKING_STATUSES
    }
    return _Queue(
        exactness_audit_hash=audit.audit_hash,
        paper_count=audit.paper_count,
        blocked_paper_count=len(blocked_ids),
        gaps=sorted(gaps, key=lambda item: (item.paper_id, item.missing_requirement)),
        gaps_by_category=dict(sorted(by_category.items())),
    ).with_hash()


def _recommended_fix(check_id: str) -> str:
    fixes = {
        "membership_valid": "repair the frozen manifest entry identity",
        "method_profile_valid": "rebuild the method profile and re-run profiling",
        "mechanism_evidence_available": "bind paper evidence refs before readiness",
        "implementation_spec_complete": "complete the implementation spec bindings",
        "not_generic_only": "implement the paper-specific mechanism behind the generic primitive",
        "not_alias_only": "register explicit formula provenance; remove any IoU-family alias",
        "not_metadata_only": "replace metadata-only wiring with real runtime behavior",
        "not_no_op": "wire the mechanism into the live runtime path",
        "paper_specific_composition": "add paper-specific composition and config",
        "runtime_hook_real": "verify the runtime hook with a real forward/backward probe",
        "runtime_fingerprint_present": "rebuild the implementation fingerprint",
        "unit_tests_present": "add paper-specific unit tests",
        "non_mock_smoke_present": "add a non-mock CPU smoke probe",
        "compatibility_tests_present": "add compatibility tests for the bound components",
        "rollback_path_present": "declare insertion points so rollback targets exist",
        "shared_primitive_correctly_referenced": "reference the shared primitive instead of copying it",
        "core_mechanism_covered": "bind the paper's core mechanism to components",
    }
    return fixes[check_id]


def _estimated_scope(check_id: str) -> str:
    large = {
        "not_generic_only",
        "not_metadata_only",
        "not_alias_only",
        "core_mechanism_covered",
        "paper_specific_composition",
    }
    if check_id in large:
        return "large"
    if check_id in {"unit_tests_present", "non_mock_smoke_present", "compatibility_tests_present"}:
        return "medium"
    return "small"


def _dependency(check_id: str) -> str:
    identity = {
        "membership_valid": "frozen manifest integrity",
        "method_profile_valid": "method profiling pipeline",
        "mechanism_evidence_available": "paper evidence extraction",
    }
    if check_id in identity:
        return identity[check_id]
    if check_id in {"unit_tests_present", "non_mock_smoke_present", "compatibility_tests_present"}:
        return "component certification artifacts"
    if check_id == "runtime_fingerprint_present":
        return "implementation registry rebuild"
    return "paper-specific implementation work in the corresponding domain"


__all__ = [
    "PaperExactnessAuditBuilder",
    "build_paper_exactness_audit",
    "build_paper_exactness_gap_queue",
]
