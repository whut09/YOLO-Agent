"""Default LLM proposal generator for loop policy planning."""

from __future__ import annotations

import json
import re
import hashlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.decision_ledger import DecisionLedger, DecisionLedgerRecord
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.agents.paper_recipe_planner import PaperRecipePlan


LLMDecisionStatus = Literal["used", "skipped", "failed"]
LLMTransport = Callable[[LLMDecisionConfig, list[dict[str, str]]], str]


class LLMDecisionAdvisorResult(BaseModel):
    """Result of an LLM decision-analysis proposal pass."""

    status: LLMDecisionStatus
    provider: str
    model: str
    model_alias: str | None = None
    prompt_sha256: str | None = None
    input_summary: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = None
    max_output_tokens: int | None = None
    proposal_bundle: "LLMProposalBundle | None" = None
    doctor_report_draft: "LLMDoctorDraft | None" = None
    evidence_requests: list["LLMEvidenceRequest"] = Field(default_factory=list)
    rejected_actions: list["LLMRejectedAction"] = Field(default_factory=list)
    proposals: list[CandidatePolicy] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw_text: str = ""


class LLMRejectedAction(BaseModel):
    """One action the LLM believes should be rejected, still audited by the harness."""

    model_config = ConfigDict(extra="forbid")

    action: str
    reason: str
    blocked_by: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class LLMEvidenceRequest(BaseModel):
    """Evidence the LLM says should be collected before stronger decisions."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    reason: str
    evidence_type: Literal[
        "metric",
        "artifact",
        "dataset_report",
        "error_fact",
        "label_quality",
        "runtime_profile",
        "other",
    ] = "other"
    target: str | None = None
    required_before: Literal["proposal", "pilot", "full", "recommendation"] = "pilot"
    priority: Literal["low", "medium", "high"] = "medium"


class LLMDoctorDraft(BaseModel):
    """Doctor-style diagnosis draft produced by the LLM for later fact alignment."""

    model_config = ConfigDict(extra="forbid")

    primary_problem: str
    likely_causes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    rejected_actions: list[LLMRejectedAction] = Field(default_factory=list)
    selected_actions: list[str] = Field(default_factory=list)
    why: list[str] = Field(default_factory=list)
    expected_improvement: dict[str, str] = Field(default_factory=dict)
    stop_condition: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class LLMProposalBundle(BaseModel):
    """Fully guarded LLM output bundle.

    The bundle is still only proposal input. Evaluators and gates decide what is
    executable.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "llm_doctor_decision.v2"
    diagnosis: str = ""
    likely_causes: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    selected_paper_priors: list[str] = Field(default_factory=list)
    selected_recipes: list[str] = Field(default_factory=list)
    rejected_paper_priors: list[str] = Field(default_factory=list)
    rejection_reasons: dict[str, str] = Field(default_factory=dict)
    implementation_requests: list[dict[str, Any]] = Field(default_factory=list)
    expected_improvement: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cost: dict[str, Any] = Field(default_factory=dict)
    stop_condition: list[str] = Field(default_factory=list)
    doctor_report_draft: LLMDoctorDraft | None = None
    evidence_requests: list[LLMEvidenceRequest] = Field(default_factory=list)
    rejected_actions: list[LLMRejectedAction] = Field(default_factory=list)
    candidate_policies: list[CandidatePolicy] = Field(default_factory=list)


class LLMComponentSelection(BaseModel):
    """One component selected from the caller-provided allowlist."""

    model_config = ConfigDict(extra="forbid")
    component_id: str
    role: str
    maturity: str
    rationale: str
    paper_prior_ids: list[str] = Field(default_factory=list)


class LLMCouplingExplanation(BaseModel):
    """Why multiple selected components cannot initially be separated."""

    model_config = ConfigDict(extra="forbid")
    component_ids: list[str] = Field(min_length=2)
    reason: str
    source_paper_ids: list[str] = Field(default_factory=list)
    internal_ablation_plan: list[str] = Field(default_factory=list)


class LLMImplementationRequest(BaseModel):
    """Non-executable request for a component that lacks a mature adapter."""

    model_config = ConfigDict(extra="forbid")
    component_id: str
    current_maturity: str
    required_adapter: str
    reason: str
    acceptance_tests: list[str] = Field(default_factory=list)


class LLMEvidenceGap(BaseModel):
    """Evidence that must be collected before a training decision."""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    reason: str
    action: Literal["import_metrics", "mine_errors", "profile_data", "advise_labels", "benchmark_latency"]
    required_before: Literal["proposal", "pilot", "full", "recommendation"] = "pilot"


class LLMRecipeEvidence(BaseModel):
    """Evidence citation with explicit local-versus-paper provenance."""

    model_config = ConfigDict(extra="forbid")
    statement: str
    source_id: str
    evidence_level: Literal["local_evidence", "paper_prior"]


class LLMPaperRecipeProposal(BaseModel):
    """Doctor-style, evidence-grounded paper recipe proposal."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "llm_paper_recipe_proposal.v1"
    primary_problem: str
    likely_causes: list[str] = Field(default_factory=list)
    evidence: list[LLMRecipeEvidence] = Field(default_factory=list)
    selected_recipe: str | None = None
    execution_action: Literal[
        "run_training", "import_metrics", "mine_errors", "profile_data",
        "advise_labels", "benchmark_latency", "implementation_only", "none",
    ] = "none"
    training_profile: Literal["debug", "pilot"] | None = None
    selected_components: list[LLMComponentSelection] = Field(default_factory=list)
    coupling: LLMCouplingExplanation | None = None
    rejected_components: list[str] = Field(default_factory=list)
    rejected_reasons: dict[str, str] = Field(default_factory=dict)
    expected_improvement: dict[str, float | str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cost: dict[str, float | str] = Field(default_factory=dict)
    stop_condition: list[str] = Field(default_factory=list)
    evidence_requests: list[LLMEvidenceGap] = Field(default_factory=list)
    implementation_requests: list[LLMImplementationRequest] = Field(default_factory=list)
    fixed_constraints: dict[str, Any] = Field(default_factory=lambda: {"imgsz": 640})

    @model_validator(mode="after")
    def _recipe_shape(self) -> "LLMPaperRecipeProposal":
        if self.fixed_constraints.get("imgsz", 640) != 640:
            raise ValueError("LLM paper recipe cannot change imgsz from 640")
        if self.training_profile not in {None, "debug", "pilot"}:
            raise ValueError("LLM paper recipe cannot request candidate_full")
        if len(self.selected_components) > 1 and self.coupling is None:
            raise ValueError("multiple selected components require coupling explanation")
        return self


class LLMPaperRecipeCriticResult(BaseModel):
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LLMPaperRecipeAdvisorResult(BaseModel):
    status: LLMDecisionStatus
    provider: str
    model: str
    prompt_sha256: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    proposal: LLMPaperRecipeProposal | None = None
    critic: LLMPaperRecipeCriticResult
    fallback_plan: PaperRecipePlan
    warnings: list[str] = Field(default_factory=list)
    raw_text: str = ""


class LLMDecisionAdvisor:
    """Generate policy proposals with an LLM, guarded by later evaluators."""

    def __init__(
        self,
        config: LLMDecisionConfig | None = None,
        transport: LLMTransport | None = None,
    ) -> None:
        self.config = config or load_llm_decision_config()
        self.transport = transport or _openai_responses_transport

    def propose(
        self,
        *,
        task_spec: TaskSpec,
        diagnosis_report: ErrorDrivenLoopReport,
        inherited_context: dict[str, Any] | None = None,
    ) -> LLMDecisionAdvisorResult:
        """Ask the configured LLM for proposals, returning structured fallback state."""
        config = self.config
        input_summary = _input_summary(task_spec, diagnosis_report, inherited_context or {})
        messages = _messages(config, task_spec, diagnosis_report, inherited_context or {})
        prompt_sha256 = _prompt_sha256(messages)
        if not config.can_generate_proposals:
            return _result(
                config,
                "skipped",
                warnings=["llm_decision_config_disabled"],
                prompt_sha256=prompt_sha256,
                input_summary=input_summary,
            )
        if _redacted(config):
            return _result(
                config,
                "skipped",
                warnings=["llm_decision_config_redacted"],
                prompt_sha256=prompt_sha256,
                input_summary=input_summary,
            )
        api_key = config.resolved_api_key()
        if config.require_api_key and not api_key:
            return _result(
                config,
                "skipped",
                warnings=[f"missing_api_key:{config.api_key_source()}"],
                prompt_sha256=prompt_sha256,
                input_summary=input_summary,
            )

        failures: list[str] = []
        raw_text = ""
        for attempt in range(config.max_retries + 1):
            try:
                raw_text = self.transport(config, messages)
                break
            except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                failures.append(f"attempt_{attempt + 1}:{exc}")
                if attempt >= config.max_retries:
                    break
                time.sleep(config.retry_backoff_seconds * (attempt + 1))
        if not raw_text:
            return _result(
                config,
                "failed",
                warnings=[f"llm_call_failed:{' | '.join(failures)}"],
                prompt_sha256=prompt_sha256,
                input_summary=input_summary,
            )

        proposal_bundle, warnings = _parse_proposal_bundle(raw_text)
        decision_context = (inherited_context or {}).get("decision_context", {})
        if proposal_bundle is not None and isinstance(decision_context, dict) and decision_context:
            proposal_bundle, guard_warnings = _guard_unified_bundle(proposal_bundle, decision_context)
            warnings.extend(guard_warnings)
        proposals = proposal_bundle.candidate_policies if proposal_bundle is not None else []
        has_structured_content = bool(
            proposal_bundle
            and (
                proposal_bundle.candidate_policies
                or proposal_bundle.doctor_report_draft
                or proposal_bundle.evidence_requests
                or proposal_bundle.rejected_actions
                or proposal_bundle.selected_paper_priors
                or proposal_bundle.selected_recipes
                or proposal_bundle.implementation_requests
            )
        )
        return LLMDecisionAdvisorResult(
            status="used" if has_structured_content else "failed",
            provider=config.provider,
            model=config.model,
            model_alias=config.model_alias,
            prompt_sha256=prompt_sha256,
            input_summary=input_summary,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            proposal_bundle=proposal_bundle,
            doctor_report_draft=proposal_bundle.doctor_report_draft if proposal_bundle is not None else None,
            evidence_requests=proposal_bundle.evidence_requests if proposal_bundle is not None else [],
            rejected_actions=proposal_bundle.rejected_actions if proposal_bundle is not None else [],
            proposals=proposals,
            warnings=warnings if has_structured_content else [*warnings, "llm_returned_no_valid_proposal_bundle"],
            raw_text=raw_text,
        )

    def propose_paper_recipe(
        self,
        *,
        top_error_facts: list[dict[str, Any]],
        paper_records: list[PaperRecord],
        component_contracts: list[ComponentContract],
        compatibility_results: dict[str, Any],
        policy_memory: list[dict[str, Any]],
        prior_pilot_deltas: list[dict[str, Any]],
        fixed_constraints: dict[str, Any],
        budget: dict[str, Any],
        available_executable_adapters: list[str],
        local_evidence: list[dict[str, Any]],
        fallback_plan: PaperRecipePlan,
        decision_ledger: DecisionLedger | None = None,
        run_id: str = "research",
    ) -> LLMPaperRecipeAdvisorResult:
        """Generate a guarded doctor-style recipe while preserving rule fallback."""
        config = self.config
        component_ids = {item.component_id for item in component_contracts}
        summary = {
            "top_unresolved_error_facts": top_error_facts,
            "selected_paper_ids": [item.paper_id for item in paper_records],
            "component_maturity": {item.component_id: item.maturity for item in component_contracts},
            "compatibility_results": compatibility_results,
            "policy_memory": policy_memory,
            "prior_pilot_deltas": prior_pilot_deltas,
            "fixed_constraints": {**fixed_constraints, "imgsz": 640},
            "budget": budget,
            "available_executable_adapters": available_executable_adapters,
            "local_evidence": local_evidence,
            "allowed_component_ids": sorted(component_ids),
            "rule_fallback": fallback_plan.model_dump(mode="json"),
        }
        messages = _paper_recipe_messages(config, summary, paper_records, component_contracts)
        prompt_sha256 = _prompt_sha256(messages)

        def finish(status: LLMDecisionStatus, proposal: LLMPaperRecipeProposal | None, critic: LLMPaperRecipeCriticResult, warnings: list[str], raw_text: str = "") -> LLMPaperRecipeAdvisorResult:
            result = LLMPaperRecipeAdvisorResult(status=status, provider=config.provider, model=config.model, prompt_sha256=prompt_sha256, input_summary=summary, proposal=proposal, critic=critic, fallback_plan=fallback_plan, warnings=warnings, raw_text=raw_text)
            _write_paper_recipe_ledger(decision_ledger, run_id, result, config)
            return result

        if not config.can_generate_proposals or _redacted(config):
            return finish("skipped", None, LLMPaperRecipeCriticResult(accepted=False, rejection_reasons=["llm_unavailable_rule_fallback"]), ["paper_recipe_llm_skipped"])
        if config.require_api_key and not config.resolved_api_key():
            return finish("skipped", None, LLMPaperRecipeCriticResult(accepted=False, rejection_reasons=["missing_api_key"]), [f"missing_api_key:{config.api_key_source()}"])

        try:
            raw_text = self.transport(config, messages)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return finish("failed", None, LLMPaperRecipeCriticResult(accepted=False, rejection_reasons=["llm_call_failed"]), [f"llm_call_failed:{exc}"])
        proposal, parse_warnings = _parse_paper_recipe(raw_text)
        if proposal is None:
            return finish("failed", None, LLMPaperRecipeCriticResult(accepted=False, rejection_reasons=["invalid_llm_paper_recipe"]), parse_warnings, raw_text)
        critic = _critique_paper_recipe(
            proposal,
            component_contracts=component_contracts,
            provided_paper_ids={item.paper_id for item in paper_records},
            available_executable_adapters=set(available_executable_adapters),
            missing_key_evidence=_missing_key_evidence(top_error_facts, local_evidence),
        )
        return finish("used" if critic.accepted else "failed", proposal, critic, parse_warnings, raw_text)


def _messages(
    config: LLMDecisionConfig,
    task_spec: TaskSpec,
    diagnosis_report: ErrorDrivenLoopReport,
    inherited_context: dict[str, Any],
) -> list[dict[str, str]]:
    system = config.prompt_contract.system_summary or (
        "You are the YOLO Agent decision-analysis model. Generate structured policy proposals only. "
        "Do not approve experiments or training directly."
    )
    payload = {
        "task_spec": task_spec.model_dump(mode="json"),
        "diagnosis_report": diagnosis_report.model_dump(mode="json"),
        "inherited_context": inherited_context,
        "diagnostic_evidence_gate": {
            "missing_diagnostic_evidence": inherited_context.get("missing_diagnostic_evidence", []),
            "llm_evidence_only_mode": bool(inherited_context.get("llm_evidence_only_mode", False)),
            "rule": (
                "When llm_evidence_only_mode is true, candidate_policies must use action_domain='evidence' "
                "and execution_action in import_metrics|mine_errors|profile_data|advise_labels|benchmark_latency. "
                "Do not output run_training proposals."
            ),
        },
        "policy_memory_context": inherited_context.get("policy_memory_context", {}),
        "decision_context": inherited_context.get("decision_context", {}),
        "decision_context_hash": inherited_context.get("decision_context_hash"),
        "output_contract": {
            "format": "JSON object",
            "schema": {
                "schema_version": "llm_doctor_decision.v2",
                "diagnosis": "single primary diagnosis",
                "likely_causes": ["causes grounded in supplied evidence"],
                "evidence": [{"source_id": "provided evidence id", "statement": "supported fact"}],
                "selected_paper_priors": ["provided prior id only"],
                "selected_recipes": ["provided recipe or policy id only"],
                "rejected_paper_priors": ["provided prior id"],
                "rejection_reasons": {"provided id": "reason"},
                "implementation_requests": [{"component_id": "provided metadata-only component", "reason": "adapter needed"}],
                "expected_improvement": {"metric": "bounded expectation, not a benchmark"},
                "confidence": 0.0,
                "cost": {"training": "low|medium|high"},
                "stop_condition": ["bounded pilot stop condition"],
                "doctor_report_draft": {
                    "primary_problem": "short diagnosis",
                    "likely_causes": ["causes grounded in provided facts"],
                    "evidence": ["facts from diagnosis_report/error facts only"],
                    "rejected_actions": [
                        {
                            "action": "action id",
                            "reason": "why rejected",
                            "blocked_by": ["constraint or guardrail names"],
                            "missing_evidence": ["needed evidence ids"],
                        }
                    ],
                    "selected_actions": ["action ids"],
                    "why": ["short reasoning"],
                    "expected_improvement": {"metric": "bounded expected range"},
                    "stop_condition": ["pilot or evidence stop condition"],
                    "missing_evidence": ["evidence ids"],
                    "confidence": "low|medium|high",
                },
                "evidence_requests": [
                    {
                        "evidence_id": "stable evidence id",
                        "reason": "why this evidence changes the decision",
                        "evidence_type": "metric|artifact|dataset_report|error_fact|label_quality|runtime_profile|other",
                        "target": "candidate/node/class/metric target",
                        "required_before": "proposal|pilot|full|recommendation",
                        "priority": "low|medium|high",
                    }
                ],
                "rejected_actions": [
                    {
                        "action": "action id",
                        "reason": "why rejected",
                        "blocked_by": ["constraint or guardrail names"],
                        "missing_evidence": ["needed evidence ids"],
                    }
                ],
                "candidate_policies": [
                    {
                        "policy_id": "llm_short_id",
                        "source": "llm",
                        "action_domain": "data|model|augmentation|postprocess|label|training|evidence",
                        "action_id": "short_action_id",
                        "execution_action": "run_training|import_metrics|mine_errors|profile_data|advise_labels|benchmark_latency",
                        "base_model": "string",
                        "scale": "n|s|m|baseline",
                        "framework": "ultralytics",
                        "components": ["component ids"],
                        "train_overrides": {},
                        "evidence_required": ["metric or artifact names"],
                        "target_error_facts": ["facts from diagnosis_report/error facts only"],
                        "expected_improvement": {"metric_name": "ap_small", "minimum_expected_delta": "pilot_positive_delta"},
                        "expected_effect": ["why this targets the diagnosis"],
                        "risk": "low|medium|high",
                        "rationale": "short explanation",
                    }
                ],
            },
        },
        "hard_rules": [
            "Return JSON only.",
            "LLM output is proposal_generator_only; never approve execution.",
            "Prefer evidence actions when required evidence is missing.",
            "If ap_small, per_class_ap/per_class_ar, or confusion_matrix evidence is missing, output evidence actions only.",
            "When diagnostic_evidence_gate.llm_evidence_only_mode is true, do not output run_training candidate_policies.",
            "Use policy_memory_context as prior experience for expected effect, cost, and risk, but never as approval to bypass guards.",
            "Prefer historically positive, low-cost actions; defer actions with negative effect, high latency cost, or low confidence unless more evidence is requested.",
            "Do not increase imgsz when guardrails require fixed baseline comparison.",
            "Do not propose candidate_full directly; propose debug/pilot-safe actions.",
            "Keep each policy to one primary variable whenever possible.",
            "Use decision_context as the canonical round input; do not create a separate paper decision path.",
            "Paper candidates, executable adapters, component maturity, compatibility, tried actions, objective, and budget in decision_context are binding context.",
            "Only propose executable training candidates that can pass the provided maturity and compatibility constraints; otherwise request evidence or implementation.",
            "Only select paper ids, prior ids, recipe ids, and component ids present in decision_context.",
            "Metadata-only components must appear only in implementation_requests.",
            "Never invent benchmarks or local evidence; cite only supplied evidence ids.",
            "Never output candidate_full or modify fixed_constraints.imgsz=640.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _input_summary(
    task_spec: TaskSpec,
    diagnosis_report: ErrorDrivenLoopReport,
    inherited_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact audit summary of the evidence visible to the LLM."""
    next_round = diagnosis_report.next_round
    memory_context = inherited_context.get("policy_memory_context", {})
    memory_effects = memory_context.get("historical_effects", []) if isinstance(memory_context, dict) else []
    return {
        "task": {
            "task_type": task_spec.task_type,
            "scene": task_spec.scene,
            "primary_metric": task_spec.primary_metric.name,
            "class_count": len(task_spec.class_names),
        },
        "diagnosis": {
            "diagnostic_count": len(diagnosis_report.diagnostics),
            "rule_policy_count": len(next_round.candidate_policies),
            "evidence_required": list(next_round.evidence_required),
            "guardrails": list(next_round.guardrails),
            "changed_variables": list(next_round.changed_variables),
        },
        "inherited_context": {
            "run_id": inherited_context.get("run_id"),
            "dataset_version": inherited_context.get("dataset_version"),
            "proposal_mode": inherited_context.get("proposal_mode"),
            "missing_diagnostic_evidence": inherited_context.get("missing_diagnostic_evidence", []),
            "llm_evidence_only_mode": bool(inherited_context.get("llm_evidence_only_mode", False)),
            "policy_memory": {
                "summary_count": memory_context.get("summary_count", 0) if isinstance(memory_context, dict) else 0,
                "effect_actions": [
                    item.get("action")
                    for item in memory_effects[:8]
                    if isinstance(item, dict) and item.get("action")
                ],
            },
            "current_round_focus": inherited_context.get("inherited_current_round_focus", []),
            "current_round_error_actions": inherited_context.get("inherited_current_round_error_actions", []),
            "guardrails": inherited_context.get("inherited_guardrails", []),
        },
    }


def _prompt_sha256(messages: list[dict[str, str]]) -> str:
    """Hash the exact prompt messages sent or prepared for the LLM."""
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _openai_responses_transport(config: LLMDecisionConfig, messages: list[dict[str, str]]) -> str:
    api_key = config.resolved_api_key()
    base_url = config.resolved_base_url()
    endpoint = (base_url.rstrip("/") if base_url else "https://api.openai.com/v1") + "/responses"
    payload = {
        "model": config.model,
        "input": messages,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return _extract_response_text(json.loads(body))


def openai_responses_transport(config: LLMDecisionConfig, messages: list[dict[str, str]]) -> str:
    """Public shared transport for explicit, pre-training LLM production jobs."""
    return _openai_responses_transport(config, messages)


def _extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return str(data["output_text"])
    output = data.get("output", [])
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.append(part["text"])
        if texts:
            return "\n".join(texts)
    choices = data.get("choices", [])
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return str(message["content"])
    return json.dumps(data, ensure_ascii=False)


def _parse_proposal_bundle(raw_text: str) -> tuple[LLMProposalBundle | None, list[str]]:
    warnings: list[str] = []
    try:
        payload = json.loads(_strip_json_fence(raw_text))
    except json.JSONDecodeError as exc:
        return None, [f"llm_json_parse_failed:{exc}"]
    if not isinstance(payload, (dict, list)):
        return None, ["llm_payload_not_mapping_or_list"]

    if isinstance(payload, list):
        raw_policies = payload
        raw_doctor = None
        raw_evidence_requests = []
        raw_rejected_actions = []
    else:
        allowed_keys = {
            "schema_version",
            "candidate_policies",
            "policies",
            "doctor_report_draft",
            "evidence_requests",
            "rejected_actions",
            "diagnosis",
            "likely_causes",
            "evidence",
            "selected_paper_priors",
            "selected_recipes",
            "rejected_paper_priors",
            "rejection_reasons",
            "implementation_requests",
            "expected_improvement",
            "confidence",
            "cost",
            "stop_condition",
        }
        unknown_keys = sorted(str(key) for key in payload if key not in allowed_keys)
        if unknown_keys:
            warnings.append(f"llm_unknown_top_level_keys:{','.join(unknown_keys)}")
        raw_policies = payload.get("candidate_policies", payload.get("policies", []))
        raw_doctor = payload.get("doctor_report_draft")
        raw_evidence_requests = payload.get("evidence_requests", [])
        raw_rejected_actions = payload.get("rejected_actions", [])

    if not isinstance(raw_policies, list):
        warnings.append("llm_candidate_policies_not_list")
        raw_policies = []
    proposals: list[CandidatePolicy] = []
    for index, item in enumerate(raw_policies):
        if not isinstance(item, dict):
            warnings.append(f"llm_policy_{index}_not_mapping")
            continue
        normalized = {
            "source": "llm",
            "base_model": "yolo26n.pt",
            "scale": "n",
            "framework": "ultralytics",
            **item,
        }
        policy_id = str(normalized.get("policy_id") or f"llm_policy_{index + 1}")
        normalized["policy_id"] = policy_id if policy_id.startswith("llm_") else f"llm_{policy_id}"
        normalized["source"] = "llm"
        try:
            proposals.append(CandidatePolicy.model_validate(normalized))
        except ValueError as exc:
            warnings.append(f"{normalized['policy_id']}:invalid_policy:{exc}")

    doctor_report_draft = _parse_doctor_draft(raw_doctor, warnings)
    evidence_requests = _parse_list_items(
        raw_evidence_requests,
        LLMEvidenceRequest,
        "evidence_request",
        warnings,
    )
    rejected_actions = _parse_list_items(
        raw_rejected_actions,
        LLMRejectedAction,
        "rejected_action",
        warnings,
    )
    try:
        return LLMProposalBundle(
            schema_version=str(payload.get("schema_version", "llm_proposal_bundle.v1"))
            if isinstance(payload, dict)
            else "llm_proposal_bundle.v1",
            doctor_report_draft=doctor_report_draft,
            evidence_requests=evidence_requests,
            rejected_actions=rejected_actions,
            candidate_policies=proposals,
            diagnosis=str(payload.get("diagnosis", "")) if isinstance(payload, dict) else "",
            likely_causes=_string_list(payload.get("likely_causes", [])) if isinstance(payload, dict) else [],
            evidence=_mapping_list(payload.get("evidence", [])) if isinstance(payload, dict) else [],
            selected_paper_priors=_string_list(payload.get("selected_paper_priors", [])) if isinstance(payload, dict) else [],
            selected_recipes=_string_list(payload.get("selected_recipes", [])) if isinstance(payload, dict) else [],
            rejected_paper_priors=_string_list(payload.get("rejected_paper_priors", [])) if isinstance(payload, dict) else [],
            rejection_reasons=dict(payload.get("rejection_reasons", {})) if isinstance(payload, dict) and isinstance(payload.get("rejection_reasons", {}), dict) else {},
            implementation_requests=_mapping_list(payload.get("implementation_requests", [])) if isinstance(payload, dict) else [],
            expected_improvement=dict(payload.get("expected_improvement", {})) if isinstance(payload, dict) and isinstance(payload.get("expected_improvement", {}), dict) else {},
            confidence=float(payload.get("confidence", 0.0)) if isinstance(payload, dict) else 0.0,
            cost=dict(payload.get("cost", {})) if isinstance(payload, dict) and isinstance(payload.get("cost", {}), dict) else {},
            stop_condition=_string_list(payload.get("stop_condition", [])) if isinstance(payload, dict) else [],
        ), warnings
    except ValueError as exc:
        return None, [*warnings, f"llm_bundle_validation_failed:{exc}"]


def _parse_doctor_draft(raw_doctor: Any, warnings: list[str]) -> LLMDoctorDraft | None:
    if raw_doctor is None:
        return None
    if not isinstance(raw_doctor, dict):
        warnings.append("llm_doctor_report_draft_not_mapping")
        return None
    try:
        return LLMDoctorDraft.model_validate(raw_doctor)
    except ValueError as exc:
        warnings.append(f"llm_doctor_report_draft_invalid:{exc}")
        return None


def _guard_unified_bundle(
    bundle: LLMProposalBundle,
    context: dict[str, Any],
) -> tuple[LLMProposalBundle, list[str]]:
    """Apply allowlists and deterministic evidence/maturity gates after one LLM call."""
    warnings: list[str] = []
    paper_candidates = _mapping_list(context.get("paper_candidates", []))
    allowed_priors = _collect_ids(paper_candidates, {"prior_id", "paper_id", "recipe_id"})
    deterministic = _mapping_list(context.get("deterministic_recipe_candidates", []))
    allowed_recipes = _collect_ids(deterministic, {"policy_id", "recipe_id"}) | _collect_ids(
        paper_candidates, {"recipe_id", "prior_id"}
    )
    maturity = context.get("component_maturity", {})
    maturity = maturity if isinstance(maturity, dict) else {}
    allowed_components = set(str(item) for item in maturity)
    selected_priors = _allowed_values(bundle.selected_paper_priors, allowed_priors, "paper_prior", warnings)
    rejected_priors = _allowed_values(bundle.rejected_paper_priors, allowed_priors, "paper_prior", warnings)
    selected_recipes = _allowed_values(bundle.selected_recipes, allowed_recipes, "recipe", warnings)
    rejected_by_critic = {
        str(item.get("recipe_id"))
        for item in _mapping_list(context.get("recipe_critic_results", []))
        if item.get("accepted") is False or item.get("decision") in {"rejected", "needs_implementation"}
    }
    selected_recipes = [
        item for item in selected_recipes
        if not _reject_recipe_critic(item, rejected_by_critic, warnings)
    ]
    implementation_requests = []
    for request in bundle.implementation_requests:
        component_id = str(request.get("component_id", ""))
        if component_id not in allowed_components:
            warnings.append(f"unknown_component_id:{component_id}")
            continue
        implementation_requests.append(request)

    missing = _string_list(context.get("missing_evidence", []))
    accepted_policies: list[CandidatePolicy] = []
    for policy in bundle.candidate_policies:
        reasons: list[str] = []
        unknown = sorted(set(policy.components) - allowed_components) if allowed_components else []
        reasons.extend(f"unknown_component_id:{item}" for item in unknown)
        metadata_only = sorted(
            item for item in policy.components if str(maturity.get(item, "metadata_only")) == "metadata_only"
        )
        if metadata_only:
            reasons.extend(f"metadata_only_component:{item}" for item in metadata_only)
            for item in metadata_only:
                if not any(str(req.get("component_id")) == item for req in implementation_requests):
                    implementation_requests.append({"component_id": item, "reason": "adapter implementation required"})
        if missing and policy.execution_action == "run_training":
            reasons.append("missing_key_evidence_blocks_run_training")
        if str(policy.action_id or "").lower().find("candidate_full") >= 0:
            reasons.append("candidate_full_forbidden")
        if "imgsz" in policy.train_overrides and str(policy.train_overrides["imgsz"]) != "640":
            reasons.append("fixed_imgsz_640_violation")
        if reasons:
            warnings.extend(f"{policy.policy_id}:{item}" for item in reasons)
        else:
            accepted_policies.append(policy)

    allowed_evidence_ids = _collect_ids(
        [
            *_mapping_list(context.get("baseline_evidence", [])),
            *_mapping_list(context.get("current_evidence", [])),
            *_mapping_list(context.get("error_facts", [])),
        ],
        {"evidence_id", "record_id", "fact_id", "node_id", "source_id"},
    )
    evidence: list[dict[str, Any]] = []
    for item in bundle.evidence:
        source_id = str(item.get("source_id", ""))
        if source_id and allowed_evidence_ids and source_id not in allowed_evidence_ids:
            warnings.append(f"invented_evidence_source:{source_id}")
            continue
        evidence.append(item)
    return bundle.model_copy(update={
        "selected_paper_priors": selected_priors,
        "selected_recipes": selected_recipes,
        "rejected_paper_priors": rejected_priors,
        "implementation_requests": implementation_requests,
        "candidate_policies": accepted_policies,
        "evidence": evidence,
    }), list(dict.fromkeys(warnings))


def _allowed_values(
    values: list[str],
    allowed: set[str],
    label: str,
    warnings: list[str],
) -> list[str]:
    accepted = []
    for value in values:
        if value in allowed:
            accepted.append(value)
        else:
            warnings.append(f"unknown_{label}_id:{value}")
    return accepted


def _reject_recipe_critic(value: str, rejected: set[str], warnings: list[str]) -> bool:
    if value not in rejected:
        return False
    warnings.append(f"recipe_critic_rejected:{value}")
    return True


def _collect_ids(items: list[dict[str, Any]], keys: set[str]) -> set[str]:
    values: set[str] = set()
    for item in items:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                values.add(value)
            elif isinstance(value, list):
                values.update(str(entry) for entry in value if entry)
        for value in item.values():
            if isinstance(value, dict):
                values.update(_collect_ids([value], keys))
            elif isinstance(value, list):
                values.update(_collect_ids([entry for entry in value if isinstance(entry, dict)], keys))
    return values


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _parse_list_items(
    raw_items: Any,
    model: type[LLMEvidenceRequest] | type[LLMRejectedAction],
    label: str,
    warnings: list[str],
) -> list[LLMEvidenceRequest] | list[LLMRejectedAction]:
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        warnings.append(f"llm_{label}s_not_list")
        return []
    parsed: list[LLMEvidenceRequest] | list[LLMRejectedAction] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            warnings.append(f"llm_{label}_{index}_not_mapping")
            continue
        try:
            parsed.append(model.model_validate(item))
        except ValueError as exc:
            warnings.append(f"llm_{label}_{index}_invalid:{exc}")
    return parsed


def _paper_recipe_messages(
    config: LLMDecisionConfig,
    summary: dict[str, Any],
    papers: list[PaperRecord],
    contracts: list[ComponentContract],
) -> list[dict[str, str]]:
    system = (
        "You are the doctor-style paper recipe advisor inside YOLO Agent. "
        "You may only select component_ids explicitly supplied by the harness. "
        "Metadata-only components are implementation requests, never executable selections. "
        "If critical evidence is missing, request evidence and do not output run_training. "
        "Keep imgsz=640 and training_profile debug or pilot; never candidate_full. "
        "Paper claims are priors and every paper-derived evidence item must use evidence_level=paper_prior."
    )
    payload = {
        "decision_context": summary,
        "paper_records": [paper.model_dump(mode="json") for paper in papers],
        "component_contracts": [contract.model_dump(mode="json") for contract in contracts],
        "output_schema": {
            "schema_version": "llm_paper_recipe_proposal.v1",
            "primary_problem": "string",
            "likely_causes": ["string"],
            "evidence": [{"statement": "string", "source_id": "paper/node/fact id", "evidence_level": "local_evidence|paper_prior"}],
            "selected_recipe": "recipe id or null",
            "execution_action": "run_training|import_metrics|mine_errors|profile_data|advise_labels|benchmark_latency|implementation_only|none",
            "training_profile": "debug|pilot|null",
            "selected_components": [{"component_id": "allowed id", "role": "string", "maturity": "provided maturity", "rationale": "string", "paper_prior_ids": ["paper id"]}],
            "coupling": {"component_ids": ["id A", "id B"], "reason": "string", "source_paper_ids": ["paper id"], "internal_ablation_plan": ["A only", "B only", "A+B"]},
            "rejected_components": ["allowed id"],
            "rejected_reasons": {"component id": "reason"},
            "expected_improvement": {"metric": "bounded range"},
            "confidence": 0.0,
            "cost": {"gpu_hours": 0.0, "risk": "low|medium|high"},
            "stop_condition": ["string"],
            "evidence_requests": [{"evidence_id": "id", "reason": "string", "action": "import_metrics|mine_errors|profile_data|advise_labels|benchmark_latency", "required_before": "proposal|pilot|full|recommendation"}],
            "implementation_requests": [{"component_id": "allowed id", "current_maturity": "maturity", "required_adapter": "adapter", "reason": "string", "acceptance_tests": ["string"]}],
            "fixed_constraints": {"imgsz": 640},
        },
    }
    configured = config.prompt_contract.system_summary.strip()
    return [
        {"role": "system", "content": f"{configured}\n\n{system}" if configured else system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _parse_paper_recipe(raw_text: str) -> tuple[LLMPaperRecipeProposal | None, list[str]]:
    try:
        payload = json.loads(_strip_json_fence(raw_text))
    except json.JSONDecodeError as exc:
        return None, [f"llm_paper_recipe_invalid_json:{exc}"]
    if not isinstance(payload, dict):
        return None, ["llm_paper_recipe_not_mapping"]
    try:
        return LLMPaperRecipeProposal.model_validate(payload), []
    except ValueError as exc:
        return None, [f"llm_paper_recipe_validation_failed:{exc}"]


def _critique_paper_recipe(
    proposal: LLMPaperRecipeProposal,
    *,
    component_contracts: list[ComponentContract],
    provided_paper_ids: set[str],
    available_executable_adapters: set[str],
    missing_key_evidence: list[str],
) -> LLMPaperRecipeCriticResult:
    contracts = {item.component_id: item for item in component_contracts}
    reasons: list[str] = []
    warnings: list[str] = []
    selected_ids = [item.component_id for item in proposal.selected_components]
    unknown = sorted(set(selected_ids) - set(contracts))
    if unknown:
        reasons.extend(f"unknown_component:{item}" for item in unknown)
    rejected_unknown = sorted(set(proposal.rejected_components) - set(contracts))
    reasons.extend(f"unknown_rejected_component:{item}" for item in rejected_unknown)
    reason_key_unknown = sorted(set(proposal.rejected_reasons) - set(contracts))
    reasons.extend(f"unknown_rejected_reason_component:{item}" for item in reason_key_unknown)
    implementation_ids = {item.component_id for item in proposal.implementation_requests}
    for selection in proposal.selected_components:
        contract = contracts.get(selection.component_id)
        if contract is None:
            continue
        if selection.maturity != contract.maturity:
            reasons.append(f"component_maturity_mismatch:{selection.component_id}")
        unknown_papers = sorted(set(selection.paper_prior_ids) - provided_paper_ids)
        reasons.extend(f"unknown_paper_prior:{paper_id}" for paper_id in unknown_papers)
        if not contract.can_execute or not contract.adapter_class or contract.adapter_class not in available_executable_adapters:
            reasons.append(f"non_executable_component_selected:{selection.component_id}")
            if selection.component_id not in implementation_ids:
                reasons.append(f"missing_implementation_request:{selection.component_id}")
    for request in proposal.implementation_requests:
        contract = contracts.get(request.component_id)
        if contract is None:
            reasons.append(f"unknown_implementation_component:{request.component_id}")
        elif contract.can_execute and contract.adapter_class in available_executable_adapters:
            warnings.append(f"implementation_request_for_executable_component:{request.component_id}")
    if missing_key_evidence and proposal.execution_action == "run_training":
        reasons.append("critical_evidence_missing_blocks_run_training")
    if missing_key_evidence and not proposal.evidence_requests:
        reasons.append("critical_evidence_missing_without_request")
    if proposal.execution_action == "run_training" and proposal.training_profile not in {"debug", "pilot"}:
        reasons.append("training_must_be_debug_or_pilot")
    if proposal.execution_action == "run_training" and not proposal.selected_components:
        reasons.append("run_training_without_selected_component")
    if proposal.fixed_constraints.get("imgsz") != 640:
        reasons.append("violates_fixed_imgsz")
    if len(selected_ids) > 1:
        if proposal.coupling is None:
            reasons.append("missing_coupling_explanation")
        elif set(proposal.coupling.component_ids) != set(selected_ids):
            reasons.append("coupling_components_mismatch")
    if proposal.coupling is not None:
        unknown_coupled = sorted(set(proposal.coupling.component_ids) - set(contracts))
        reasons.extend(f"unknown_coupling_component:{item}" for item in unknown_coupled)
        unknown_coupling_papers = sorted(set(proposal.coupling.source_paper_ids) - provided_paper_ids)
        reasons.extend(f"unknown_coupling_paper:{item}" for item in unknown_coupling_papers)
    for item in proposal.evidence:
        if item.source_id in provided_paper_ids and item.evidence_level != "paper_prior":
            reasons.append(f"paper_claim_not_marked_prior:{item.source_id}")
    return LLMPaperRecipeCriticResult(accepted=not reasons, rejection_reasons=list(dict.fromkeys(reasons)), warnings=list(dict.fromkeys(warnings)))


def _missing_key_evidence(error_facts: list[dict[str, Any]], local_evidence: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    if not error_facts:
        missing.append("top_unresolved_error_facts")
    verified = [item for item in local_evidence if bool(item.get("verified", True))]
    if not verified:
        missing.append("verified_local_evidence")
    return missing


def _write_paper_recipe_ledger(
    ledger: DecisionLedger | None,
    run_id: str,
    result: LLMPaperRecipeAdvisorResult,
    config: LLMDecisionConfig,
) -> None:
    if ledger is None:
        return
    ledger.append(DecisionLedgerRecord(
        run_id=run_id,
        policy_id=result.proposal.selected_recipe if result.proposal and result.proposal.selected_recipe else "llm_paper_recipe",
        decision_type="llm_paper_recipe",
        proposal={
            "output": result.proposal.model_dump(mode="json") if result.proposal else None,
            "critic": result.critic.model_dump(mode="json"),
            "fallback_plan": result.fallback_plan.model_dump(mode="json"),
        },
        decision="accepted" if result.critic.accepted else "rule_fallback",
        prompt_sha256=result.prompt_sha256,
        input_summary=result.input_summary,
        model_metadata={"provider": config.provider, "model": config.model, "temperature": config.temperature, "status": result.status, "warnings": result.warnings},
        blocked_by=result.critic.rejection_reasons,
        errors=result.warnings,
        rationale="Doctor-style paper recipe proposal; deterministic critic and rule fallback remain authoritative.",
        policy_version="llm_paper_recipe.v1",
    ))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _redacted(config: LLMDecisionConfig) -> bool:
    return "XX" in {config.provider, config.model} or (
        config.api_key_env == "XX" and not config.api_key
    )


def _result(
    config: LLMDecisionConfig,
    status: LLMDecisionStatus,
    warnings: list[str] | None = None,
    prompt_sha256: str | None = None,
    input_summary: dict[str, Any] | None = None,
) -> LLMDecisionAdvisorResult:
    return LLMDecisionAdvisorResult(
        status=status,
        provider=config.provider,
        model=config.model,
        model_alias=config.model_alias,
        prompt_sha256=prompt_sha256,
        input_summary=input_summary or {},
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        warnings=warnings or [],
    )


__all__ = [
    "LLMComponentSelection",
    "LLMCouplingExplanation",
    "LLMDecisionAdvisor",
    "LLMDecisionAdvisorResult",
    "LLMDoctorDraft",
    "LLMEvidenceGap",
    "LLMEvidenceRequest",
    "LLMImplementationRequest",
    "LLMPaperRecipeAdvisorResult",
    "LLMPaperRecipeCriticResult",
    "LLMPaperRecipeProposal",
    "LLMProposalBundle",
    "LLMRecipeEvidence",
    "LLMRejectedAction",
    "openai_responses_transport",
]
