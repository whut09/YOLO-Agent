"""Certified paper recipe materialization into ASHA-owned plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from yolo_agent.agents.decision_bundle import DecisionContext
from yolo_agent.agents.paper_candidate_orchestrator import (
    PaperCandidateOrchestrator,
    PaperCandidateSubmission,
)
from yolo_agent.agents.paper_component_gate import (
    PaperComponentEligibilityGate,
    PaperEligibilityBudget,
    PaperEligibilityConstraints,
)
from yolo_agent.agents.paper_recipe_materialization.evidence import (
    current_materialization_error_facts,
    evidence_recovery_for_facts,
)
from yolo_agent.agents.paper_recipe_materialization.candidate_priority import (
    rank_materialized_candidate,
)
from yolo_agent.agents.paper_recipe_materialization.matched_control import (
    assess_matched_control,
)
from yolo_agent.agents.paper_recipe_materialization.requests import (
    implementation_request_from_materialization,
    runtime_implementation_request,
)
from yolo_agent.agents.paper_recipe_materialization.runtime_identity import (
    certified_runtime_identity,
)
from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperRecipeCandidateGateResult,
    PaperRecipeCandidateInput,
    PaperRecipeMaterializationResult,
)
from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverageLedger,
    PaperProposalDisposition,
)
from yolo_agent.agents.recipe_critic import RecipeCritic
from yolo_agent.components.adapters import ComponentAdapterRegistry
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.certification.component_queue_gate import (
    ComponentQueueCertificationGate,
)
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.policy_memory import PolicyMemoryRecord
from yolo_agent.recipes.recipe_materializer import RecipeMaterializer
from yolo_agent.research.snapshot import ResearchSnapshot


class PaperRecipeMaterializationGate:
    """Only certified local adapters may cross from paper prior into ASHA."""

    policy_version = "paper_recipe_materialization_gate.v1"

    def __init__(
        self,
        run_dir: Path | str,
        *,
        base_run_id: str,
        adapter_registry: ComponentAdapterRegistry | None = None,
        orchestrator: PaperCandidateOrchestrator | None = None,
        certification_report_path: Path | str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_registry = adapter_registry or ComponentAdapterRegistry()
        self.orchestrator = orchestrator or PaperCandidateOrchestrator(
            self.run_dir,
            base_run_id=base_run_id,
        )
        self.ledger = DecisionLedger(self.run_dir / "artifacts" / "decision_ledger.jsonl")
        self.materializer = RecipeMaterializer()
        self.critic = RecipeCritic()
        self.eligibility_gate = PaperComponentEligibilityGate(self.ledger)
        self.execution_bridge = ComponentExecutionBridge(
            adapter_registry=self.adapter_registry,
        )
        self.component_certification_gate = ComponentQueueCertificationGate()
        self.certification_report_path = (
            Path(certification_report_path)
            if certification_report_path is not None
            else None
        )

    def _proposal_ledger(
        self,
        *,
        run_id: str,
        protocol_hash: str,
    ) -> PaperCandidateCoverageLedger:
        return PaperCandidateCoverageLedger(
            self.run_dir / "artifacts" / "paper_candidate_coverage.yaml",
            run_id=run_id,
            protocol_hash=protocol_hash,
        )

    def materialize(
        self,
        *,
        run_id: str,
        decision_context: DecisionContext,
        research_snapshot: ResearchSnapshot,
        candidates: Iterable[PaperRecipeCandidateInput],
        current_error_facts: Iterable[ErrorFact],
        component_contracts: Mapping[str, ComponentContract],
        objective: OptimizationObjective,
        budget: PaperEligibilityBudget,
        round_index: int,
        local_evidence: Iterable[PolicyMemoryRecord | dict[str, Any]] = (),
    ) -> PaperRecipeMaterializationResult:
        candidate_inputs = list(candidates)
        facts = list(current_error_facts)
        proposal_ledger = self._proposal_ledger(
            run_id=run_id,
            protocol_hash=objective.baseline_protocol_hash,
        )
        input_fingerprints = {
            item.prior.prior_id: _materialization_fingerprint(item, objective.baseline_protocol_hash)
            for item in candidate_inputs
        }
        proposal_ledger.upsert_many(
            [
                _materialization_input_record(
                    item,
                    run_id=run_id,
                    round_index=round_index,
                    execution_fingerprint=input_fingerprints[item.prior.prior_id],
                )
                for item in candidate_inputs
            ]
        )

        def mark(
            item: PaperRecipeCandidateInput,
            disposition: str,
            reasons: list[str],
            source_stage: str,
        ) -> None:
            reasons = list(dict.fromkeys(reasons)) or [
                "stage_transition_without_reason"
            ]
            proposal_ledger.update_disposition(
                execution_fingerprint=input_fingerprints[item.prior.prior_id],
                disposition=disposition,  # type: ignore[arg-type]
                reason_codes=list(dict.fromkeys(reasons)),
                source_stage=source_stage,
                candidate_id=(
                    item.source_node.candidate_config.candidate_id
                    if item.source_node is not None
                    else None
                ),
                node_id=item.source_node.node_id if item.source_node is not None else None,
                required_evidence=(
                    list(reasons) if disposition == "evidence_recovery" else None
                ),
                required_adapters=(
                    list(item.prior.required_adapter)
                    if disposition == "implementation_request"
                    else None
                ),
            )

        recovery = evidence_recovery_for_facts(
            facts,
            run_id=run_id,
            protocol_hash=objective.baseline_protocol_hash,
        )
        if recovery is not None:
            for item in candidate_inputs:
                mark(item, "evidence_recovery", recovery.required_evidence, "materialization")
            self._record_boundary(
                run_id,
                decision="evidence_recovery",
                reason=recovery.reason,
                missing_evidence=recovery.required_evidence,
            )
            return PaperRecipeMaterializationResult(
                run_id=run_id,
                action="evidence_recovery",
                evidence_recovery=recovery,
                stopped_reason="current_protocol_error_facts_missing",
                terminal_lines=[
                    "Paper recipes: evidence recovery only",
                    "Training: blocked until current-node COCO error facts are complete",
                    "Budget authority: ASHA (no assignment created)",
                ],
                ledger_path=proposal_ledger.path,
            )

        snapshot_error = _snapshot_error(decision_context, research_snapshot)
        if snapshot_error:
            for item in candidate_inputs:
                mark(item, "blocked_runtime", [snapshot_error], "materialization")
            self._record_boundary(run_id, decision="blocked", reason=snapshot_error)
            return PaperRecipeMaterializationResult(
                run_id=run_id,
                action="blocked",
                stopped_reason=snapshot_error,
                terminal_lines=[f"Paper recipes: blocked ({snapshot_error})"],
                ledger_path=proposal_ledger.path,
            )
        if not candidate_inputs:
            return self._exhausted(run_id, "no_certified_paper_components")

        current = current_materialization_error_facts(
            facts,
            run_id=run_id,
            protocol_hash=objective.baseline_protocol_hash,
        )
        evidence = list(local_evidence)
        submissions: list[PaperCandidateSubmission] = []
        outcomes: list[PaperRecipeCandidateGateResult] = []
        for item in candidate_inputs:
            prior = item.prior
            profile_errors = _method_profile_errors(item)
            if profile_errors:
                mark(item, "incompatible", profile_errors, "materialization")
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="rejected",
                    reasons=profile_errors,
                ))
                continue
            materialized = self.materializer.materialize(
                prior,
                component_contracts=component_contracts,
            )
            if materialized.recipe is None or materialized.allowed_stage != "pilot":
                mark(
                    item,
                    "implementation_request",
                    list(materialized.reasons),
                    "materialization",
                )
                request = implementation_request_from_materialization(prior, materialized)
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="implementation_request",
                    reasons=list(materialized.reasons),
                    implementation_request=request,
                ))
                continue
            recipe = materialized.recipe
            selected_contracts = {
                component_id: component_contracts[component_id]
                for component_id in recipe.component_ids
                if component_id in component_contracts
            }
            component_certification = self.component_certification_gate.evaluate(
                component_ids=list(recipe.component_ids),
                report_path=self.certification_report_path,
                component_contracts=selected_contracts,
            )
            if not component_certification.allowed:
                mark(
                    item,
                    "blocked_runtime",
                    list(component_certification.blockers),
                    "materialization",
                )
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="rejected",
                    recipe_id=recipe.recipe_id,
                    reasons=list(component_certification.blockers),
                ))
                continue
            adapters, adapter_errors = self._resolve_adapters(selected_contracts)
            if adapter_errors:
                mark(item, "implementation_request", adapter_errors, "materialization")
                request = runtime_implementation_request(prior, adapter_errors)
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="implementation_request",
                    recipe_id=recipe.recipe_id,
                    reasons=adapter_errors,
                    implementation_request=request,
                ))
                continue
            if item.source_node is None:
                mark(item, "blocked_runtime", ["candidate_source_node_missing"], "materialization")
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="rejected",
                    recipe_id=recipe.recipe_id,
                    reasons=["candidate_source_node_missing"],
                ))
                continue
            runtime = self.execution_bridge.prepare(
                recipe=recipe,
                node=item.source_node,
                contracts=selected_contracts,
                workspace=(
                    self.run_dir
                    / "artifacts"
                    / "paper_runtime"
                    / _safe_name(prior.prior_id)
                ),
                run_id=run_id,
                protocol_hash=objective.baseline_protocol_hash,
                dry_run=True,
            )
            if runtime.status != "executable":
                if "distillation.yolo26_teacher_student" in recipe.component_ids:
                    disposition = _distillation_runtime_disposition(runtime.blocked_by)
                    mark(
                        item,
                        disposition,
                        list(runtime.blocked_by),
                        "materialization",
                    )
                    outcomes.append(PaperRecipeCandidateGateResult(
                        prior_id=prior.prior_id,
                        action="rejected",
                        recipe_id=recipe.recipe_id,
                        reasons=list(runtime.blocked_by),
                    ))
                else:
                    mark(item, "implementation_request", list(runtime.blocked_by), "materialization")
                    request = runtime_implementation_request(prior, runtime.blocked_by)
                    outcomes.append(PaperRecipeCandidateGateResult(
                        prior_id=prior.prior_id,
                        action="implementation_request",
                        recipe_id=recipe.recipe_id,
                        reasons=list(runtime.blocked_by),
                        implementation_request=request,
                    ))
                continue
            try:
                identity = certified_runtime_identity(runtime)
            except ValueError as exc:
                reasons = [f"runtime_identity_rejected:{exc}"]
                mark(item, "implementation_request", reasons, "materialization")
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="implementation_request",
                    recipe_id=recipe.recipe_id,
                    reasons=reasons,
                    implementation_request=runtime_implementation_request(prior, reasons),
                ))
                continue
            priority = rank_materialized_candidate(
                item,
                current_error_facts=current,
                local_evidence=evidence,
                runtime_execution_ready=identity.runtime_execution_ready,
            )
            control = assess_matched_control(
                runtime.node,
                item.matched_control_node,
                required_protocol_hash=objective.baseline_protocol_hash,
            )
            constraints = PaperEligibilityConstraints(
                imgsz=640,
                matched_baseline=control.available,
                matched_baseline_protocol_hash=control.protocol_hash,
                research_snapshot_hash=research_snapshot.snapshot_hash,
                candidate_metrics_source="local_verified",
            )
            eligibility = self.eligibility_gate.evaluate(
                run_id=run_id,
                recipe=recipe,
                component_contracts=selected_contracts,
                component_adapters=adapters,
                compatibility=item.compatibility,
                maturity={key: value.maturity for key, value in selected_contracts.items()},
                fixed_constraints=constraints,
                research_snapshot=research_snapshot,
                current_error_facts=current,
                objective=objective,
                budget=budget,
            )
            critic = self.critic.critique(
                recipe,
                error_facts=current,
                component_contracts=selected_contracts.values(),
                compatibility={
                    component_id: {
                        "compatible": item.compatibility.ok,
                        "blocked_by": item.compatibility.errors,
                    }
                    for component_id in recipe.component_ids
                },
                local_evidence=evidence,
            )
            reasons = [*control.reasons, *eligibility.blocked_by, *critic.blocked_by]
            if reasons or not eligibility.eligible or not critic.accepted:
                mark(item, "incompatible", list(dict.fromkeys(reasons)), "materialization")
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="rejected",
                    candidate_id=runtime.node.candidate_config.candidate_id,
                    recipe_id=recipe.recipe_id,
                    reasons=list(dict.fromkeys(reasons)),
                    runtime_identity=identity,
                    eligibility_token=eligibility.eligibility_token,
                    planning_priority=priority,
                ))
                continue
            submission = PaperCandidateSubmission(
                decision_context=decision_context,
                research_snapshot=research_snapshot,
                recipe_prior=prior,
                recipe=recipe,
                eligibility=eligibility,
                critic=critic,
                runtime_identity=identity,
                source_node=runtime.node,
                matched_control_node=item.matched_control_node,
                component_family=item.component_family,
                bucket=item.bucket,
                round_index=round_index,
                method_profile_ids=[item.method_profile.profile_id],
                planning_priority=priority,
            )
            submissions.append(submission)
            mark(item, "queued", ["materialization_candidate_ready"], "materialization")
            outcomes.append(PaperRecipeCandidateGateResult(
                prior_id=prior.prior_id,
                action="registered_with_asha",
                candidate_id=runtime.node.candidate_config.candidate_id,
                recipe_id=recipe.recipe_id,
                runtime_identity=identity,
                eligibility_token=eligibility.eligibility_token,
                planning_priority=priority,
            ))

        if not submissions:
            if any(item.action == "implementation_request" for item in outcomes):
                action = "implementation_required"
            else:
                action = "exhausted"
            reason = "no_certified_paper_components"
            self._record_boundary(run_id, decision=action, reason=reason)
            return PaperRecipeMaterializationResult(
                run_id=run_id,
                action=action,
                candidates=outcomes,
                stopped_reason=reason,
                terminal_lines=[
                    f"Paper recipes: {action}",
                    f"Stop: {reason}; scalar HPO is disabled",
                    *_rejection_terminal_lines(candidate_inputs, outcomes),
                    "Training: no ASHA assignment created",
                ],
                ledger_path=proposal_ledger.path,
            )

        submissions.sort(key=_submission_priority_key)
        registration = self.orchestrator.register_cohort(submissions)
        for outcome in outcomes:
            item = next(
                candidate
                for candidate in candidate_inputs
                if candidate.prior.prior_id == outcome.prior_id
            )
            if outcome.candidate_id in registration.rejected:
                outcome.action = "rejected"
                outcome.reasons.append(registration.rejected[outcome.candidate_id])
                mark(item, "blocked_runtime", outcome.reasons, "asha_registration")
            elif outcome.candidate_id in registration.deferred:
                outcome.action = "deferred"
                outcome.reasons.append(registration.deferred[outcome.candidate_id])
                mark(item, "deferred_budget", outcome.reasons, "asha_registration")
            elif outcome.candidate_id in registration.deferred_allocation:
                outcome.action = "deferred"
                outcome.reasons.append(
                    registration.deferred_allocation_reasons.get(
                        outcome.candidate_id,
                        "deferred_by_exploit_explore_budget",
                    )
                )
                mark(
                    item,
                    "deferred_budget",
                    outcome.reasons,
                    "asha_registration",
                )
            elif outcome.candidate_id in registration.registered:
                mark(item, "queued", ["asha_trial_registered"], "asha_registration")
        step = self.orchestrator.next_step()
        if step.round_plan is not None:
            for node in step.round_plan.deferred_nodes:
                item = next(
                    (
                        candidate
                        for candidate in candidate_inputs
                        if candidate.source_node is not None
                        and candidate.source_node.candidate_config.candidate_id
                        == node.candidate_config.candidate_id
                    ),
                    None,
                )
                if item is not None:
                    mark(item, "queued", ["round_execution_plan_written"], "round_execution_plan")
        action = {
            "queue_assignment": "queue_assignment",
            "awaiting_pilot_3_cohort": "awaiting_cohort",
        }.get(step.action, "registered_with_asha")
        selected_submission = next(
            (
                submission
                for submission in submissions
                if submission.source_node.candidate_config.candidate_id
                in registration.registered
            ),
            submissions[0],
        )
        return PaperRecipeMaterializationResult(
            run_id=run_id,
            action=action,
            candidates=outcomes,
            registration=registration.model_dump(mode="json"),
            round_execution_plan=(
                step.round_plan.model_dump(mode="json") if step.round_plan is not None else None
            ),
            execution_queue=(step.queue.model_dump(mode="json") if step.queue is not None else None),
            asha_assignment_id=(
                step.assignment.assignment_id if step.assignment is not None else None
            ),
            terminal_lines=_terminal_lines(
                step.adapter_identity,
                step.reason,
                paper_ids=selected_submission.recipe_prior.paper_ids,
                priority=selected_submission.planning_priority,
            ),
            ledger_path=proposal_ledger.path,
        )

    def _resolve_adapters(
        self,
        contracts: Mapping[str, ComponentContract],
    ) -> tuple[dict[str, Any], list[str]]:
        adapters: dict[str, Any] = {}
        errors: list[str] = []
        for component_id, contract in contracts.items():
            try:
                adapters[component_id] = self.adapter_registry.create_for_contract(contract)
            except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"runtime_adapter_missing:{component_id}:{exc}")
        return adapters, errors

    def _exhausted(self, run_id: str, reason: str) -> PaperRecipeMaterializationResult:
        self._record_boundary(run_id, decision="exhausted", reason=reason)
        return PaperRecipeMaterializationResult(
            run_id=run_id,
            action="exhausted",
            stopped_reason=reason,
            terminal_lines=[
                "Paper recipes: exhausted",
                f"Stop: {reason}; scalar HPO is disabled",
                "Scalar HPO: disabled",
                "Training: no ASHA assignment created",
            ],
            ledger_path=self.ledger.path,
        )

    def _record_boundary(
        self,
        run_id: str,
        *,
        decision: str,
        reason: str,
        missing_evidence: list[str] | None = None,
    ) -> None:
        self.ledger.append(DecisionLedgerRecord(
            run_id=run_id,
            policy_id="paper_recipe_materialization",
            decision_type="paper_recipe_materialization_boundary",
            proposal={
                "scalar_hpo_enabled": False,
                "queue_authority": "ASHA/RoundExecutionPlan",
            },
            decision=decision,
            missing_evidence=missing_evidence or [],
            rationale=reason,
            policy_version=self.policy_version,
        ))


def _materialization_fingerprint(
    item: PaperRecipeCandidateInput,
    protocol_hash: str,
) -> str:
    payload = {
        "prior_id": item.prior.prior_id,
        "paper_ids": sorted(item.prior.paper_ids),
        "component_ids": sorted(item.prior.component_ids),
        "baseline_protocol": item.prior.baseline_protocol,
        "protocol_hash": protocol_hash,
        "source_candidate_id": (
            item.source_node.candidate_config.candidate_id
            if item.source_node is not None
            else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _materialization_input_record(
    item: PaperRecipeCandidateInput,
    *,
    run_id: str,
    round_index: int,
    execution_fingerprint: str,
) -> PaperProposalDisposition:
    fact_ids = [
        hashlib.sha256(
            json.dumps(fact, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        for fact in item.prior.target_error_facts
    ]
    return PaperProposalDisposition(
        run_id=run_id,
        round_index=round_index,
        paper_ids=sorted(set(item.prior.paper_ids)),
        method_profile_ids=[item.method_profile.profile_id],
        recipe_id=item.prior.prior_id,
        recipe_version=item.prior.schema_version,
        canonical_component_ids=sorted(set(item.prior.component_ids)),
        combination_fingerprint=execution_fingerprint,
        coupling_reason=item.prior.coupling_reason,
        coupling_source_papers=sorted(set(item.prior.paper_ids)),
        internal_ablation_plan=list(item.prior.internal_ablation_plan),
        execution_fingerprint=execution_fingerprint,
        candidate_id=(
            item.source_node.candidate_config.candidate_id
            if item.source_node is not None
            else None
        ),
        node_id=item.source_node.node_id if item.source_node is not None else None,
        source_stage="materialization_input",
        disposition="queued",
        reason_codes=["materialization_candidate_seen"],
        required_adapters=sorted(set(item.prior.required_adapter)),
        matched_error_fact_ids=fact_ids,
    )


def _snapshot_error(context: DecisionContext, snapshot: ResearchSnapshot) -> str:
    if not snapshot.frozen:
        return "research_snapshot_not_frozen"
    if snapshot.paper_intelligence != "available":
        return f"paper_intelligence_unavailable:{snapshot.unavailable_reason or 'unknown'}"
    if not context.research_snapshot_verified:
        return "decision_context_snapshot_not_verified"
    if context.research_snapshot_hash != snapshot.snapshot_hash:
        return "decision_context_snapshot_mismatch"
    return ""


def _distillation_runtime_disposition(blockers: Iterable[str]) -> str:
    values = list(blockers)
    recoverable = (
        "checkpoint_missing",
        "sha256_missing",
        "teacher checkpoint",
    )
    return (
        "evidence_recovery"
        if any(any(marker in blocker for marker in recoverable) for blocker in values)
        else "blocked_runtime"
    )


def _method_profile_errors(item: PaperRecipeCandidateInput) -> list[str]:
    """Keep paper provenance and implementation routing bound to the prior."""
    profile = item.method_profile
    decision = item.implementation_decision
    prior = item.prior
    errors: list[str] = []
    if profile.paper_id not in prior.paper_ids:
        errors.append("paper_method_profile_paper_mismatch")
    if profile.profile_id != decision.profile_id or profile.paper_id != decision.paper_id:
        errors.append("paper_method_profile_decision_mismatch")
    if set(profile.canonical_component_ids) != set(prior.component_ids):
        errors.append("paper_method_profile_component_mismatch")
    if set(decision.canonical_component_ids) != set(prior.component_ids):
        errors.append("paper_implementation_decision_component_mismatch")
    if decision.decision not in {"reuse_existing_adapter", "coupled_recipe"}:
        errors.append(
            f"paper_implementation_decision_not_trainable:{decision.decision}"
        )
    if decision.exact_reproduction_claim and decision.component_adaptation:
        errors.append("exact_reproduction_and_component_adaptation_mixed")
    return errors


def _terminal_lines(
    identity: dict[str, Any],
    reason: str,
    *,
    paper_ids: list[str] | None = None,
    priority: Any | None = None,
) -> list[str]:
    if not identity:
        return ["Paper recipes: registered with ASHA", f"State: {reason}"]
    adapters = ", ".join(
        f"{component}={identity['adapter_classes'][component]}@"
        f"{identity['adapter_versions'][component]}"
        for component in identity["adapter_ids"]
    )
    lines = [
        f"Paper: {', '.join(paper_ids or ['unknown'])}",
        f"Component: {', '.join(identity['adapter_ids'])}",
        f"Adapter: {adapters}",
        "Adapter hash: "
        + ", ".join(
            f"{component}={identity['adapter_hashes'][component]}"
            for component in identity["adapter_ids"]
        ),
        "Maturity: "
        + ", ".join(
            f"{component}={identity['component_maturity'][component]}"
            for component in identity["adapter_ids"]
        ),
        f"Adapter patch: {identity['adapter_patch_hash']}",
        f"Runtime payload: {identity['adapter_runtime_payload_hash']}",
        "Budget authority: ASHA",
        f"State: {reason}",
    ]
    if priority is not None:
        lines.insert(
            -2,
            "Planning priority: "
            f"score={priority.score:.6f} "
            f"covered_papers={priority.covered_paper_count} "
            f"mechanism_confidence={priority.canonical_mechanism_confidence:.6f}",
        )
    return lines


def _submission_priority_key(submission: PaperCandidateSubmission) -> tuple[float, str]:
    priority = submission.planning_priority
    return (
        -(priority.score if priority is not None else submission.recipe_prior.confidence),
        submission.recipe_prior.prior_id,
    )


def _rejection_terminal_lines(
    candidates: list[PaperRecipeCandidateInput],
    outcomes: list[PaperRecipeCandidateGateResult],
) -> list[str]:
    by_prior = {item.prior.prior_id: item for item in candidates}
    lines: list[str] = []
    for outcome in outcomes[:4]:
        candidate = by_prior.get(outcome.prior_id)
        if candidate is None:
            continue
        reason = "; ".join(outcome.reasons) or "not_eligible"
        lines.append(
            "Rejected: paper_id="
            f"{','.join(candidate.prior.paper_ids)} component_id="
            f"{','.join(candidate.prior.component_ids)} reason={reason}"
        )
    lines.append("Scalar HPO: disabled")
    return lines


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


__all__ = ["PaperRecipeMaterializationGate"]
