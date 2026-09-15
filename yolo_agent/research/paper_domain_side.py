"""Paper-specific distillation/domain-adaptation routing and CPU behavior audit.

The frozen paper plan is the only source of campaign membership.  Generic
teacher-student and domain-alignment branches are treated strictly as shared
primitives: a paper becomes ready only through its own paper-specific route
(``distillation.<method>`` / ``domain_adaptation.<method>``), its own config,
and its own synthetic behavior evidence.  Required teacher/checkpoint/data
assets are recorded with ``blocked_for_real_reproduction`` flags that never
block code-level readiness when the first training run can produce them.
No training is started here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.distillation.method_registry import (
    NAMED_PAPER_BRANCHES as DISTILLATION_NAMED_PAPER_BRANCHES,
)
from yolo_agent.components.adapters.distillation.paper_routes import (
    build_paper_route,
)
from yolo_agent.components.adapters.domain_adaptation.branches import (
    DOMAIN_BRANCH_PROFILES,
    NAMED_PAPER_BRANCHES as DOMAIN_NAMED_PAPER_BRANCHES,
)
from yolo_agent.components.adapters.domain_adaptation.domain_paper_routes import (
    build_domain_paper_route,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest
from yolo_agent.research.paper_83_implementation_schemas import (
    Paper83EngineeringPlan,
    Paper83EngineeringPlanEntry,
)
from yolo_agent.research.paper_domain_side_probes import (
    ProbeOutcome,
    probe_domain_branch_route,
    probe_distillation_mechanism,
)
from yolo_agent.research.paper_domain_side_schemas import (
    DomainSideBehaviorEvidence,
    DomainSideRouteSpec,
    PaperDomainSideAudit,
    PaperDomainSideRecord,
    RequiredAssetRecord,
)

DOMAIN_SIDE_DOMAINS = frozenset({"distillation", "domain_adaptation", "semi_supervised"})

_DISTILLATION_BRANCH_TO_MECHANISM = {
    "logits_distillation": "logits",
    "feature_distillation": "feature",
    "relation_distillation": "relation",
    "localization_distillation": "localization",
    "attention_distillation": "attention",
    "masked_feature_distillation": "masked_feature",
    "quality_aware_distillation": "quality_aware",
    "teacher_ensemble": "teacher_ensemble",
    "source_free_teacher": "source_free_teacher",
    "cross_domain_teacher": "cross_domain_teacher",
    "contrastive_distillation": "contrastive",
}


def _asset_kind(asset_id: str) -> str:
    lowered = asset_id.lower()
    if "teacher_checkpoint" in lowered or lowered.startswith("teacher:"):
        return "teacher_checkpoint"
    if "student" in lowered and "checkpoint" in lowered:
        return "student_checkpoint"
    if "source_model" in lowered or "source_trained" in lowered:
        return "source_checkpoint"
    if "pseudo_label" in lowered:
        return "pseudo_label_manifest"
    if "query" in lowered:
        return "query_manifest"
    if "pair" in lowered or "contrastive" in lowered:
        return "pair_manifest"
    if "target" in lowered and ("manifest" in lowered or "data" in lowered or "split" in lowered):
        return "target_domain_data"
    if "source" in lowered and ("manifest" in lowered or "data" in lowered or "split" in lowered):
        return "source_domain_data"
    if "unlabeled" in lowered:
        return "unlabeled_data"
    return "protocol_evidence"


def _recovery_action(asset_id: str, paper_id: str) -> str:
    kind = _asset_kind(asset_id)
    if kind == "teacher_checkpoint":
        return (
            "produce or provide the frozen teacher checkpoint in the first "
            f"training run for {paper_id}; the algorithm runs on synthetic "
            "teachers today"
        )
    if kind == "student_checkpoint":
        return "provide the YOLO26n student checkpoint at training start"
    if kind == "source_checkpoint":
        return (
            "train or provide the source-domain model; source-free adaptation "
            "starts from it in the first target run"
        )
    if kind in {"unlabeled_data", "target_domain_data", "source_domain_data"}:
        return "bind the dataset manifest and sha256 before the first run"
    if kind in {"pseudo_label_manifest", "query_manifest", "pair_manifest"}:
        return "generate the manifest from a frozen teacher before the first run"
    return "attach the protocol evidence artifact during readiness confirmation"


def _required_asset_records(entry: Paper83EngineeringPlanEntry) -> list[RequiredAssetRecord]:
    records: list[RequiredAssetRecord] = []
    seen: set[str] = set()
    for asset_id in entry.required_assets:
        if asset_id in seen:
            continue
        seen.add(asset_id)
        kind = _asset_kind(asset_id)
        # No required asset is bound at code-audit time: checkpoints, data
        # manifests, and protocol evidence are all produced or attached by the
        # first training run.  The algorithm itself runs on synthetic stand-ins
        # today, so the flag records the real-reproduction gap without blocking
        # code-level implementation readiness.
        records.append(
            RequiredAssetRecord(
                asset_id=asset_id,
                asset_kind=kind,  # type: ignore[arg-type]
                available_at_runtime=False,
                blocked_for_real_reproduction=True,
                recovery_action=_recovery_action(asset_id, entry.paper_id),
            )
        )
    return records


def _behavior_evidence(outcome: ProbeOutcome) -> DomainSideBehaviorEvidence:
    return DomainSideBehaviorEvidence(
        passed=outcome.passed,
        checks=dict(outcome.checks),
        observed_changes=list(outcome.observed_changes),
        errors=list(outcome.errors),
    )


def _resolve_distillation_route(paper_id: str) -> DomainSideRouteSpec | None:
    branch_id = DISTILLATION_NAMED_PAPER_BRANCHES.get(paper_id)
    route = build_paper_route(paper_id)
    mechanism_key = (
        _DISTILLATION_BRANCH_TO_MECHANISM.get(branch_id) if branch_id is not None else None
    )
    config_schema: dict[str, Any] = {}
    if branch_id is not None:
        config_schema = {
            "branch_id": branch_id,
            "mechanism": mechanism_key,
        }
    return DomainSideRouteSpec(
        mechanism_id=route.paper_specific_mechanism_id,
        route_kind="distillation_route",
        paper_id=paper_id,
        branch_id=branch_id,
        method_identity_status=route.method_identity_status,
        component_id=route.component_id,
        adapter_class=route.adapter_class,
        runtime_hooks=["build_criterion", "compute_loss"],
        required_assets=[
            "teacher_checkpoint",
            "teacher_checkpoint_sha256",
            "student:yolo26n.pt",
            "teacher_student_same_split",
        ],
        config_schema=config_schema,
    )


def _resolve_domain_route(paper_id: str) -> DomainSideRouteSpec | None:
    branch_id = DOMAIN_NAMED_PAPER_BRANCHES.get(paper_id)
    if branch_id is None:
        return None
    route = build_domain_paper_route(paper_id)
    profile = DOMAIN_BRANCH_PROFILES[branch_id]
    return DomainSideRouteSpec(
        mechanism_id=route.paper_specific_mechanism_id,
        route_kind="domain_adaptation_route",
        paper_id=paper_id,
        branch_id=branch_id,
        method_identity_status="branch_bound",
        component_id=route.component_id,
        adapter_class=route.adapter_class,
        runtime_strategy=str(profile["runtime_strategy"]),
        runtime_hooks=["build_criterion", "compute_loss"],
        required_assets=list(profile["required_evidence"]),
        config_schema={
            "branch_id": branch_id,
            "runtime_strategy": str(profile["runtime_strategy"]),
            "adaptation_mode": str(profile["adaptation_mode"]),
            "requires_source_domain": bool(profile["requires_source_domain"]),
        },
    )


class PaperDomainSideAuditBuilder:
    """Audit teacher/domain scope and behavior for every frozen paper."""

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
    ) -> PaperDomainSideAudit:
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
            raise ValueError("domain-side plan membership differs from frozen paper manifest")
        if engineering_plan.manifest_membership_hash != frozen.campaign.membership_hash:
            raise ValueError("domain-side plan membership hash differs from frozen manifest")

        by_id = {item.paper_id: item for item in frozen.papers}
        entries = {item.paper_id: item for item in engineering_plan.papers}
        if set(entries) != set(frozen_ids):
            raise ValueError("domain-side plan must contain every frozen paper exactly once")
        records = [
            self._record(by_id[paper_id], entries[paper_id])
            for paper_id in frozen_ids
        ]
        summary = {
            status: sum(item.status == status for item in records)
            for status in ("ready", "blocked_missing_evidence", "out_of_scope")
        }
        audit = PaperDomainSideAudit(
            manifest_path=str(self.manifest_path.resolve()),
            plan_path=str(self.requested_plan_path.resolve()),
            plan_source_path=str(source_plan.resolve()),
            manifest_membership_hash=frozen.campaign.membership_hash,
            plan_identity_hash=(
                engineering_plan.plan_identity_hash or engineering_plan.calculate_hash()
            ),
            paper_count=len(records),
            domain_side_paper_count=sum(item.in_scope for item in records),
            distillation_paper_count=sum(item.primary_domain == "distillation" for item in records),
            domain_adaptation_paper_count=(
                sum(item.primary_domain == "domain_adaptation" for item in records)
            ),
            semi_supervised_paper_count=sum(item.primary_domain == "semi_supervised" for item in records),
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
    ) -> PaperDomainSideRecord:
        in_scope = entry.implementation_domain in DOMAIN_SIDE_DOMAINS
        mechanisms = _unique(entry.paper_specific_mechanism_ids)
        evidence_refs = _unique([*entry.source_locations, *entry.required_evidence])
        dependencies = _unique(
            [*entry.paper_specific_missing_parts, *entry.dependency_papers_or_primitives]
        )
        if not in_scope:
            behavior = DomainSideBehaviorEvidence(
                passed=True,
                checks={"domain_side_scope": False},
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
        if not mechanisms:
            blockers.append("blocked_missing_evidence:paper_specific_domain_mechanism")
        if not entry.source_locations:
            blockers.append("blocked_missing_evidence:paper_domain_source_location")
        if not entry.required_evidence:
            blockers.append("blocked_missing_evidence:paper_domain_evidence_ref")
        if not entry.paper_specific_config:
            blockers.append("blocked_missing_evidence:paper_specific_domain_config")

        routes: list[DomainSideRouteSpec] = []
        for paper_id in [entry.paper_id]:
            if entry.implementation_domain == "distillation":
                route = _resolve_distillation_route(paper_id)
            else:
                route = _resolve_domain_route(paper_id)
            if route is None:
                blockers.append(
                    f"blocked_missing_evidence:domain_side_route_unresolved:{paper_id}"
                )
            else:
                routes.append(route)
                # Generic primitives must never be mistaken for the paper route.
                if route.method_identity_status == "identity_recovery":
                    blockers.append(
                        "blocked_missing_evidence:generic_branch_not_paper_route:"
                        + route.mechanism_id
                    )

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
        self, routes: list[DomainSideRouteSpec]
    ) -> DomainSideBehaviorEvidence:
        checks: dict[str, bool | str | int | float] = {}
        changes: list[str] = []
        errors: list[str] = []
        for route in routes:
            try:
                if route.route_kind == "distillation_route":
                    branch_id = route.branch_id
                    mechanism_key = _DISTILLATION_BRANCH_TO_MECHANISM.get(branch_id or "")
                    if mechanism_key is None:
                        errors.append(
                            f"{route.mechanism_id}:no_behavior_probe_for_identity_recovery"
                        )
                        continue
                    result = probe_distillation_mechanism(mechanism_key)
                else:
                    branch_id = route.branch_id or ""
                    if branch_id not in DOMAIN_BRANCH_PROFILES:
                        errors.append(f"{route.mechanism_id}:unknown_branch:{branch_id}")
                        continue
                    result = probe_domain_branch_route(branch_id)
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
        return DomainSideBehaviorEvidence(
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
        routes: list[DomainSideRouteSpec],
        evidence_refs: list[str],
        dependencies: list[str],
        behavior: DomainSideBehaviorEvidence,
        status: str,
        blockers: list[str],
    ) -> PaperDomainSideRecord:
        route_ids = [item.mechanism_id for item in routes]
        component_ids = _unique(item.component_id for item in routes)
        adapter_classes = _unique(item.adapter_class for item in routes)
        branch_ids = _unique(item.branch_id for item in routes if item.branch_id)
        identity_statuses = _unique(item.method_identity_status for item in routes)
        config = {
            "paper_specific_config": dict(entry.paper_specific_config),
            "routes": [item.model_dump(mode="json") for item in routes],
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
        return PaperDomainSideRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            year=paper.year,
            method_profile_id=entry.method_profile_id or "",
            current_disposition=entry.current_disposition or "unknown",
            primary_domain=entry.implementation_domain,
            in_scope=in_scope,
            paper_specific_mechanism_ids=mechanisms,
            resolved_route_ids=route_ids,
            component_ids=component_ids,
            adapter_classes=adapter_classes,
            branch_ids=branch_ids,
            method_identity_statuses=identity_statuses,
            evidence_refs=evidence_refs,
            remaining_dependencies=dependencies,
            paper_specific_config=config if in_scope else {},
            required_assets=_required_asset_records(entry) if in_scope else [],
            behavior=behavior,
            status=status,  # type: ignore[arg-type]
            blockers=blockers,
            implementation_fingerprint=fingerprint,
        )


def _is_domain_side_mechanism(value: str) -> bool:
    return value.startswith("distillation.") or value.startswith("domain_adaptation.")


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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _md(value: str) -> str:
    return value.replace("|", "\\|")


def build_paper_domain_side_audit(**kwargs: object) -> PaperDomainSideAudit:
    """Functional builder for offline callers."""

    return PaperDomainSideAuditBuilder(**kwargs).build()


def render_paper_83_domain_side_status(audit: PaperDomainSideAudit) -> str:
    """Render all 83 papers without turning out-of-scope rows into claims."""

    lines = [
        "# Paper-83 Distillation/Domain-Adaptation Status",
        "",
        "This report audits teacher, domain-adaptation, and semi-supervised",
        "mechanisms. It does not train a model. Generic teacher-student and",
        "domain-alignment branches are shared primitives; each paper's status",
        "is bound to its own paper-specific route and behavior evidence.",
        "",
        f"- Frozen papers: {audit.paper_count}",
        f"- Teacher/domain papers in scope: {audit.domain_side_paper_count}",
        f"- Distillation papers: {audit.distillation_paper_count}",
        f"- Domain-adaptation papers: {audit.domain_adaptation_paper_count}",
        f"- Semi-supervised papers: {audit.semi_supervised_paper_count}",
        f"- Ready domain-side routes: {audit.summary.get('ready', 0)}",
        f"- Blocked for missing evidence: {audit.summary.get('blocked_missing_evidence', 0)}",
        f"- Out of scope: {audit.summary.get('out_of_scope', 0)}",
        f"- Manifest membership hash: `{audit.manifest_membership_hash}`",
        f"- Plan source: `{audit.plan_source_path}`",
        "",
        "## Per-paper Audit",
        "",
        "| Paper | Domain | Scope | Status | Routes | Branch | Assets blocked | Blockers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in audit.records:
        routes = "<br>".join(item.resolved_route_ids) or "none"
        branches = "<br>".join(item.branch_ids) or "none"
        blocked_assets = sum(
            entry.blocked_for_real_reproduction for entry in item.required_assets
        )
        blockers = "<br>".join(item.blockers) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(item.paper_id)}`",
                    _md(item.primary_domain),
                    "domain-side" if item.in_scope else "out-of-scope",
                    item.status,
                    _md(routes),
                    _md(branches),
                    str(blocked_assets) if item.in_scope else "-",
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
            "- Required teacher/checkpoint/data assets are recorded per paper;",
            "  `blocked_for_real_reproduction=true` marks assets the first training",
            "  run must provide. It never blocks code-level implementation readiness",
            "  because every mechanism runs on synthetic teachers/students in the",
            "  probe suite. No in-scope paper depends on an unobtainable fixed",
            "  external model.",
            "- Behavior evidence comes from real CPU probes: mechanism forward and",
            "  backward without teacher gradients, EMA decay arithmetic, integer",
            "  buffer copy, pseudo-label confidence filtering with fail-closed",
            "  empty-set behavior, and gradient-reversal direction (forward",
            "  passthrough, negated scaled backward).",
            "- Readiness is per paper and per route: an 11-mechanism shared",
            "  distillation library or 8-strategy domain library passing probes",
            "  never marks a paper ready without its own route, config, and",
            "  evidence. Identity-recovery routes (no certified branch) are",
            "  blocked, not silently promoted.",
            "- No training is started by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_83_domain_side_status(
    audit: PaperDomainSideAudit,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_paper_83_domain_side_status(audit), encoding="utf-8")
    return path
