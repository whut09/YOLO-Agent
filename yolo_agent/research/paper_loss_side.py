"""Paper-specific loss/assignment routing and CPU behavior audit.

The frozen paper plan is the only source of campaign membership and domain
scope.  This module routes only exact paper-specific mechanism IDs to the
concrete loss plugin or assigner plugin named by the formula provenance
registry, probes real behavior (math matrix, baseline difference) on CPU, and
records per-paper readiness that never collapses into shared component
coverage.  No training is started.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


from yolo_agent.components.assignment import (
    AssignerInputs,
    build_yolo26_assigner_plugin,
    compare_assignments,
)
from yolo_agent.components.auxiliary_losses import build_auxiliary_loss
from yolo_agent.loss_math.formula_provenance import (
    ASSIGNMENT_FORMULA_PROVENANCE,
    LOSS_FORMULA_PROVENANCE,
)
from yolo_agent.loss_math.loss_math_matrix import evaluate_loss_math_matrix
from yolo_agent.loss_math.tal_reference import (
    ppyoloe_tal_profile,
    tal_reference_assignment,
    yolo_baseline_profile,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_implementation_schemas import (
    Paper83EngineeringPlan,
    Paper83EngineeringPlanEntry,
)
from yolo_agent.research.paper_loss_side_schemas import (
    LossSideBehaviorEvidence,
    LossSideRouteSpec,
    PaperLossSideAudit,
    PaperLossSideRecord,
)

LOSS_SIDE_DOMAINS = frozenset(
    {
        "bbox_loss",
        "classification_loss",
        "auxiliary_loss",
        "assignment",
        "quality_alignment",
    }
)


def _probe_auxiliary_loss(mechanism_id: str) -> LossSideBehaviorEvidence:
    """Run the shared math matrix plus baseline-difference property checks."""

    provenance = LOSS_FORMULA_PROVENANCE[mechanism_id]
    plugin_name = provenance.implementation_name
    results = evaluate_loss_math_matrix(plugin_name, build_loss=build_auxiliary_loss)
    errors: list[str] = []
    checks: dict[str, bool | str | int | float] = {}
    for result in results:
        checks[f"{result.case}.passed"] = result.passed
        checks[f"{result.case}.finite_output"] = result.finite_output
        checks[f"{result.case}.finite_gradient"] = result.finite_gradient
        checks[f"{result.case}.backward_ran"] = result.backward_ran
        if not result.passed:
            errors.append(f"{result.case}: {'; '.join(result.errors)}")
    finite_gradient_cases = sum(1 for result in results if result.finite_gradient)
    checks["math_matrix_cases"] = len(results)
    checks["math_matrix_autograd_cases"] = finite_gradient_cases
    return LossSideBehaviorEvidence(
        passed=not errors,
        checks=checks,
        observed_changes=[],
        errors=errors,
    )


def _assignment_fixture() -> AssignerInputs:
    from tests.assignment_fixtures import assignment_inputs

    return assignment_inputs()


def _probe_assignment(mechanism_id: str) -> LossSideBehaviorEvidence:
    """Prove the paper assigner differs from the task-aligned baseline.

    ``assigner.task_aligned`` is the PP-YOLOE paper profile of the shared TAL
    metric core: the probe therefore compares the paper configuration
    (alpha=1.0, beta=6.0, topk=13, alignment-normalized soft targets) against
    the YOLOv8-default baseline configuration of the same core.
    """

    provenance = ASSIGNMENT_FORMULA_PROVENANCE[mechanism_id]
    method = provenance.implementation_method
    inputs = _assignment_fixture()
    checks: dict[str, bool | str | int | float] = {}
    errors: list[str] = []
    changes: list[str] = []

    baseline = tal_reference_assignment(inputs, yolo_baseline_profile())
    if method == "tood_tal":
        candidate = tal_reference_assignment(inputs, ppyoloe_tal_profile())
    else:
        candidate = build_yolo26_assigner_plugin(method).run(inputs)
    comparison = compare_assignments(baseline, candidate)
    checks["foreground_disagreement_count"] = comparison.foreground_disagreement_count
    checks["candidate_positive_count"] = comparison.candidate_positive_count
    checks["baseline_positive_count"] = comparison.baseline_positive_count
    if comparison.foreground_disagreement_count == 0:
        errors.append("paper assigner did not change the assignment vs baseline")
    else:
        changes.append("foreground_mask")
    if comparison.gt_conflict_count > 0:
        changes.append("target_gt_indices")
    if method == "tood_tal":
        # PP-YOLOE soft targets are normalized alignment values in (0, 1].
        positive_scores = candidate.target_scores[0][candidate.foreground_mask[0]]
        positive_labels = candidate.target_labels[0][candidate.foreground_mask[0]]
        quality = positive_scores.gather(-1, positive_labels.unsqueeze(-1).long()).squeeze(-1)
        checks["normalized_soft_target_max"] = float(quality.max())
        checks["normalized_soft_target_min"] = float(quality.min())
        if not (float(quality.max()) <= 1.0 and float(quality.min()) > 0.0):
            errors.append("pp-yoloe soft targets are not alignment-normalized")
    checks["plugin_positive_ratio"] = comparison.candidate_positive_ratio
    return LossSideBehaviorEvidence(
        passed=not errors,
        checks=checks,
        observed_changes=changes,
        errors=errors,
    )


def resolve_loss_side_route(mechanism_id: str) -> LossSideRouteSpec | None:
    """Resolve an exact mechanism ID to its provenance-registered route."""

    if mechanism_id in LOSS_FORMULA_PROVENANCE:
        provenance = LOSS_FORMULA_PROVENANCE[mechanism_id]
        return LossSideRouteSpec(
            mechanism_id=mechanism_id,
            route_kind="auxiliary_loss",
            component_id=f"loss.aux.{provenance.implementation_name}",
            adapter_id=f"loss.aux.{provenance.insertion_point}"
            if hasattr(provenance, "insertion_point")
            else f"loss.aux.{provenance.implementation_name}",
            implementation_path="yolo_agent/components/auxiliary_losses.py",
            plugin_name=provenance.implementation_name,
            runtime_hooks=[f"criterion.aux_loss.{provenance.implementation_name}"],
            required_evidence=[provenance.source_anchor],
            test_refs=[
                "tests/test_loss_math_properties.py",
                "tests/test_loss_math_matrix.py",
            ],
            config_schema={"imgsz": 640},
        )
    if mechanism_id in ASSIGNMENT_FORMULA_PROVENANCE:
        provenance = ASSIGNMENT_FORMULA_PROVENANCE[mechanism_id]
        return LossSideRouteSpec(
            mechanism_id=mechanism_id,
            route_kind="assignment",
            component_id=f"assigner.{provenance.implementation_method}",
            adapter_id=f"assigner.{provenance.implementation_method}",
            implementation_path="yolo_agent/components/assignment.py",
            plugin_name=provenance.implementation_method,
            runtime_hooks=[f"criterion.assigner.{provenance.implementation_method}"],
            required_evidence=[provenance.source_anchor],
            test_refs=[
                "tests/test_assignment_algorithms.py",
                "tests/test_assignment_paper_difference.py",
            ],
            config_schema={"imgsz": 640},
        )
    return None


class PaperLossSideAuditBuilder:
    """Audit loss/assignment scope and behavior for every frozen paper."""

    def __init__(
        self,
        *,
        manifest_path: Path | str = "configs/research/paper_83_manifest.yaml",
        plan_path: Path | str = "configs/research/paper_83_implementation_plan.yaml",
        workspace: Path | str = ".",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.requested_plan_path = Path(plan_path)
        self.workspace = Path(workspace).resolve()

    def build(
        self,
        *,
        manifest: Paper83Manifest | None = None,
        plan: Paper83EngineeringPlan | None = None,
    ) -> PaperLossSideAudit:
        """Build a complete audit without starting a trainer."""

        frozen = manifest or Paper83Manifest.from_yaml(self.manifest_path)
        if frozen.paper_count != 83:
            raise ValueError(
                f"frozen paper manifest must contain 83 papers, got {frozen.paper_count}"
            )
        engineering_plan, source_plan = self._load_plan(plan)
        frozen_ids = [item.paper_id for item in frozen.papers]
        plan_ids = list(engineering_plan.manifest_paper_ids) or [
            item.paper_id for item in engineering_plan.papers
        ]
        if plan_ids != frozen_ids:
            raise ValueError("loss-side plan membership differs from frozen paper manifest")
        if engineering_plan.manifest_membership_hash != frozen.campaign.membership_hash:
            raise ValueError("loss-side plan membership hash differs from frozen manifest")

        by_id = {item.paper_id: item for item in frozen.papers}
        entries = {item.paper_id: item for item in engineering_plan.papers}
        if set(entries) != set(frozen_ids):
            raise ValueError("loss-side plan must contain every frozen paper exactly once")
        records = [
            self._record(by_id[paper_id], entries[paper_id])
            for paper_id in frozen_ids
        ]
        summary = {
            status: sum(item.status == status for item in records)
            for status in ("ready", "blocked_missing_evidence", "out_of_scope")
        }
        audit = PaperLossSideAudit(
            manifest_path=str(self.manifest_path.resolve()),
            plan_path=str(self.requested_plan_path.resolve()),
            plan_source_path=str(source_plan.resolve()),
            manifest_membership_hash=frozen.campaign.membership_hash,
            plan_identity_hash=(
                engineering_plan.plan_identity_hash
                or engineering_plan.calculate_hash()
            ),
            paper_count=len(records),
            loss_side_paper_count=sum(item.in_scope for item in records),
            records=sorted(records, key=lambda item: item.paper_id),
            summary=summary,
        )
        return audit.with_hash()

    def _load_plan(
        self,
        plan: Paper83EngineeringPlan | None,
    ) -> tuple[Paper83EngineeringPlan, Path]:
        if plan is not None:
            return plan, self.requested_plan_path
        if self.requested_plan_path.is_file():
            return (
                Paper83EngineeringPlan.from_yaml(self.requested_plan_path),
                self.requested_plan_path,
            )
        if self.requested_plan_path.name == "paper_83_implementation_plan.yaml":
            fallback = self.requested_plan_path.with_name("paper_83_engineering_plan.yaml")
            if fallback.is_file():
                return (
                    Paper83EngineeringPlan.from_yaml(fallback),
                    fallback,
                )
        raise FileNotFoundError(
            f"paper-83 implementation plan does not exist: {self.requested_plan_path}"
        )

    def _record(
        self,
        paper: Any,
        entry: Paper83EngineeringPlanEntry,
    ) -> PaperLossSideRecord:
        domains = {entry.implementation_domain, *entry.secondary_domains}
        in_scope = bool(domains.intersection(LOSS_SIDE_DOMAINS))
        mechanisms = _unique(entry.paper_specific_mechanism_ids)
        evidence_refs = _unique([*entry.source_locations, *entry.required_evidence])
        dependencies = _unique(
            [*entry.paper_specific_missing_parts, *entry.dependency_papers_or_primitives]
        )
        if not in_scope:
            behavior = LossSideBehaviorEvidence(
                passed=True,
                checks={"loss_side_scope": False},
            )
            return self._record_value(
                paper=paper,
                entry=entry,
                in_scope=False,
                mechanisms=[],
                routes=[],
                evidence_refs=evidence_refs,
                dependencies=dependencies,
                behavior=behavior,
                status="out_of_scope",
                blockers=[],
            )

        blockers: list[str] = []
        if entry.evidence_status != "sufficient_for_planning":
            blockers.append(
                "blocked_missing_evidence:plan_evidence_status=" + entry.evidence_status
            )
        if not entry.source_locations:
            blockers.append("blocked_missing_evidence:paper_loss_source_location")
        if not entry.required_evidence:
            blockers.append("blocked_missing_evidence:paper_loss_evidence_ref")
        if not entry.paper_specific_config:
            blockers.append("blocked_missing_evidence:paper_specific_loss_config")

        routes: list[LossSideRouteSpec] = []
        for mechanism_id in mechanisms:
            if not _is_loss_mechanism(mechanism_id):
                continue
            route = resolve_loss_side_route(mechanism_id)
            if route is None:
                blockers.append(
                    f"blocked_missing_evidence:loss_side_route_unresolved:{mechanism_id}"
                )
            else:
                routes.append(route)

        if not routes and not blockers:
            blockers.append("blocked_missing_evidence:no_loss_side_route")
        behavior = self._probe_routes(routes)
        if not behavior.passed:
            blockers.extend(
                f"blocked_missing_evidence:behavior:{error}" for error in behavior.errors
            )
        blockers = _unique(blockers)
        status = "ready" if not blockers else "blocked_missing_evidence"
        return self._record_value(
            paper=paper,
            entry=entry,
            in_scope=True,
            mechanisms=mechanisms,
            routes=routes,
            evidence_refs=evidence_refs,
            dependencies=dependencies,
            behavior=behavior,
            status=status,
            blockers=blockers,
        )

    def _probe_routes(
        self, routes: list[LossSideRouteSpec]
    ) -> LossSideBehaviorEvidence:
        checks: dict[str, bool | str | int | float] = {}
        changes: list[str] = []
        errors: list[str] = []
        for route in routes:
            try:
                if route.route_kind == "auxiliary_loss":
                    result = _probe_auxiliary_loss(route.mechanism_id)
                else:
                    result = _probe_assignment(route.mechanism_id)
            except (ImportError, RuntimeError, TypeError, ValueError, KeyError) as exc:
                errors.append(f"{route.mechanism_id}:{type(exc).__name__}:{exc}")
                continue
            checks.update(
                {f"{route.mechanism_id}.{key}": value for key, value in result.checks.items()}
           
            )
            changes.extend(
                f"{route.mechanism_id}:{change}" for change in result.observed_changes
            )
            errors.extend(f"{route.mechanism_id}:{error}" for error in result.errors)
        return LossSideBehaviorEvidence(
            passed=not errors
            and all(bool(value) for value in checks.values() if isinstance(value, bool)),
            checks=checks,
            observed_changes=sorted(set(changes)),
            errors=sorted(set(errors)),
        )

    def _record_value(
        self,
        *,
        paper: Any,
        entry: Paper83EngineeringPlanEntry,
        in_scope: bool,
        mechanisms: list[str],
        routes: list[LossSideRouteSpec],
        evidence_refs: list[str],
        dependencies: list[str],
        behavior: LossSideBehaviorEvidence,
        status: str,
        blockers: list[str],
    ) -> PaperLossSideRecord:
        route_ids = [item.mechanism_id for item in routes]
        component_ids = _unique(item.component_id for item in routes if item.component_id)
        adapter_ids = _unique(item.adapter_id for item in routes if item.adapter_id)
        config = {
            "paper_specific_config": dict(entry.paper_specific_config),
            "routes": [item.model_dump(mode="json") for item in routes],
            "fixed_imgsz": 640,
            "source_locations": evidence_refs,
        }
        fingerprint = _hash_payload(
            {
                "paper_id": paper.paper_id,
                "method_profile_id": entry.method_profile_id,
                "mechanisms": mechanisms,
                "routes": config,
            }
        )
        return PaperLossSideRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            method_profile_id=paper.method_profile_id or "",
            current_disposition=paper.current_disposition or "unknown",
            primary_domain=entry.implementation_domain,
            secondary_domains=list(entry.secondary_domains),
            in_scope=in_scope,
            paper_specific_mechanism_ids=mechanisms,
            resolved_route_ids=route_ids,
            component_ids=component_ids,
            adapter_ids=adapter_ids,
            evidence_refs=evidence_refs,
            remaining_dependencies=dependencies,
            paper_specific_config=config if in_scope else {},
            behavior=behavior,
            status=status,  # type: ignore[arg-type]
            blockers=blockers,
            implementation_fingerprint=fingerprint,
        )


def _is_loss_mechanism(value: str) -> bool:
    return value in LOSS_FORMULA_PROVENANCE or value in ASSIGNMENT_FORMULA_PROVENANCE


def _unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _hash_payload(payload: Any) -> str:
    import hashlib

    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _md(value: str) -> str:
    return value.replace("|", "\\|")


def build_paper_loss_side_audit(**kwargs: object) -> PaperLossSideAudit:
    """Functional builder for offline callers."""

    return PaperLossSideAuditBuilder(**kwargs).build()


def render_paper_83_loss_side_status(audit: PaperLossSideAudit) -> str:
    """Render all 83 papers without turning out-of-scope rows into claims."""

    lines = [
        "# Paper-83 Loss/Assignment-side Status",
        "",
        "This report audits loss, assignment, and quality-alignment mechanisms. "
        "It does not train a model and does not promote shared primitives to "
        "paper implementations.",
        "",
        f"- Frozen papers: {audit.paper_count}",
        f"- Loss/assignment papers in scope: {audit.loss_side_paper_count}",
        f"- Ready loss-side routes: {audit.summary.get('ready', 0)}",
        f"- Blocked for missing evidence: {audit.summary.get('blocked_missing_evidence', 0)}",
        f"- Out of scope: {audit.summary.get('out_of_scope', 0)}",
        f"- Manifest membership hash: `{audit.manifest_membership_hash}`",
        f"- Plan source: `{audit.plan_source_path}`",
        "",
        "## Per-paper Audit",
        "",
        "| Paper | Domain | Scope | Status | Mechanisms | Routes | Blockers |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in audit.records:
        mechanisms = "<br>".join(item.paper_specific_mechanism_ids) or "none"
        routes = "<br>".join(item.resolved_route_ids) or "none"
        blockers = "<br>".join(item.blockers) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(item.paper_id)}`",
                    _md(item.primary_domain),
                    "loss-side" if item.in_scope else "out-of-scope",
                    item.status,
                    _md(mechanisms),
                    _md(routes),
                    _md(blockers),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Every routed loss carries a formula-provenance record with a source"
            " anchor, parameter semantics, shared primitives, and distinguishing"
            " terms that guard against CIoU-weight aliasing.",
            "- Every routed assigner carries an original-vs-YOLO26 contract record"
            " (adapter transform, preserved and approximated information) and is"
            " labeled `faithful_adaptation`.",
            "- Loss behavior evidence comes from a shared degenerate-input math"
            " matrix (finite output, autograd backward, finite gradients) plus"
            " paper-specific property tests.",
            "- Assignment behavior evidence comes from a pure-torch task-aligned"
            " baseline reference compared with the paper assigner on constructed"
            " inputs; identical assignments fail the audit.",
            "- Readiness is per paper: a shared class passing tests never marks a"
            " paper ready without its own composition, config, and evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_83_loss_side_status(
    audit: PaperLossSideAudit,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_paper_83_loss_side_status(audit), encoding="utf-8")
    return path
