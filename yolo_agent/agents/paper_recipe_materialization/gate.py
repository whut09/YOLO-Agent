"""Certified paper recipe materialization into ASHA-owned plans."""

from __future__ import annotations

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
        recovery = evidence_recovery_for_facts(
            facts,
            run_id=run_id,
            protocol_hash=objective.baseline_protocol_hash,
        )
        if recovery is not None:
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
                ledger_path=self.ledger.path,
            )

        snapshot_error = _snapshot_error(decision_context, research_snapshot)
        if snapshot_error:
            self._record_boundary(run_id, decision="blocked", reason=snapshot_error)
            return PaperRecipeMaterializationResult(
                run_id=run_id,
                action="blocked",
                stopped_reason=snapshot_error,
                terminal_lines=[f"Paper recipes: blocked ({snapshot_error})"],
                ledger_path=self.ledger.path,
            )
        if not candidate_inputs:
            return self._exhausted(run_id, "paper_component_recipes_exhausted")

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
            materialized = self.materializer.materialize(
                prior,
                component_contracts=component_contracts,
            )
            if materialized.recipe is None or materialized.allowed_stage != "pilot":
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
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="rejected",
                    recipe_id=recipe.recipe_id,
                    reasons=list(component_certification.blockers),
                ))
                continue
            adapters, adapter_errors = self._resolve_adapters(selected_contracts)
            if adapter_errors:
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
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="implementation_request",
                    recipe_id=recipe.recipe_id,
                    reasons=reasons,
                    implementation_request=runtime_implementation_request(prior, reasons),
                ))
                continue
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
                outcomes.append(PaperRecipeCandidateGateResult(
                    prior_id=prior.prior_id,
                    action="rejected",
                    candidate_id=runtime.node.candidate_config.candidate_id,
                    recipe_id=recipe.recipe_id,
                    reasons=list(dict.fromkeys(reasons)),
                    runtime_identity=identity,
                    eligibility_token=eligibility.eligibility_token,
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
            )
            submissions.append(submission)
            outcomes.append(PaperRecipeCandidateGateResult(
                prior_id=prior.prior_id,
                action="registered_with_asha",
                candidate_id=runtime.node.candidate_config.candidate_id,
                recipe_id=recipe.recipe_id,
                runtime_identity=identity,
                eligibility_token=eligibility.eligibility_token,
            ))

        if not submissions:
            if any(item.action == "implementation_request" for item in outcomes):
                action = "implementation_required"
                reason = "no_certified_paper_runtime_adapter_available"
            else:
                action = "exhausted"
                reason = "paper_component_recipes_exhausted"
            self._record_boundary(run_id, decision=action, reason=reason)
            return PaperRecipeMaterializationResult(
                run_id=run_id,
                action=action,
                candidates=outcomes,
                stopped_reason=reason,
                terminal_lines=[
                    f"Paper recipes: {action}",
                    f"Stop: {reason}; scalar HPO is disabled",
                    "Training: no ASHA assignment created",
                ],
                ledger_path=self.ledger.path,
            )

        registration = self.orchestrator.register_cohort(submissions)
        for outcome in outcomes:
            if outcome.candidate_id in registration.rejected:
                outcome.action = "rejected"
                outcome.reasons.append(registration.rejected[outcome.candidate_id])
            elif outcome.candidate_id in registration.deferred:
                outcome.action = "deferred"
                outcome.reasons.append(registration.deferred[outcome.candidate_id])
        step = self.orchestrator.next_step()
        action = {
            "queue_assignment": "queue_assignment",
            "awaiting_pilot_3_cohort": "awaiting_cohort",
        }.get(step.action, "registered_with_asha")
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
            terminal_lines=_terminal_lines(step.adapter_identity, step.reason),
            ledger_path=self.ledger.path,
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


def _terminal_lines(identity: dict[str, Any], reason: str) -> list[str]:
    if not identity:
        return ["Paper recipes: registered with ASHA", f"State: {reason}"]
    adapters = ", ".join(
        f"{component}={identity['adapter_classes'][component]}@"
        f"{identity['adapter_versions'][component]}"
        for component in identity["adapter_ids"]
    )
    return [
        f"Adapter: {adapters}",
        f"Adapter patch: {identity['adapter_patch_hash']}",
        f"Runtime payload: {identity['adapter_runtime_payload_hash']}",
        "Budget authority: ASHA",
        f"State: {reason}",
    ]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


__all__ = ["PaperRecipeMaterializationGate"]
