"""Default LLM proposal generator for loop policy planning."""

from __future__ import annotations

import json
import os
import re
import hashlib
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.core.llm_config import LLMDecisionConfig, load_llm_decision_config
from yolo_agent.core.task_spec import TaskSpec


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

    schema_version: str = "llm_proposal_bundle.v1"
    doctor_report_draft: LLMDoctorDraft | None = None
    evidence_requests: list[LLMEvidenceRequest] = Field(default_factory=list)
    rejected_actions: list[LLMRejectedAction] = Field(default_factory=list)
    candidate_policies: list[CandidatePolicy] = Field(default_factory=list)


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
        api_key = os.environ.get(config.api_key_env)
        if config.require_api_key and not api_key:
            return _result(
                config,
                "skipped",
                warnings=[f"missing_api_key_env:{config.api_key_env}"],
                prompt_sha256=prompt_sha256,
                input_summary=input_summary,
            )

        try:
            raw_text = self.transport(config, messages)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return _result(
                config,
                "failed",
                warnings=[f"llm_call_failed:{exc}"],
                prompt_sha256=prompt_sha256,
                input_summary=input_summary,
            )

        proposal_bundle, warnings = _parse_proposal_bundle(raw_text)
        proposals = proposal_bundle.candidate_policies if proposal_bundle is not None else []
        has_structured_content = bool(
            proposal_bundle
            and (
                proposal_bundle.candidate_policies
                or proposal_bundle.doctor_report_draft
                or proposal_bundle.evidence_requests
                or proposal_bundle.rejected_actions
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
        "output_contract": {
            "format": "JSON object",
            "schema": {
                "schema_version": "llm_proposal_bundle.v1",
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
            "Do not increase imgsz when guardrails require fixed baseline comparison.",
            "Do not propose candidate_full directly; propose debug/pilot-safe actions.",
            "Keep each policy to one primary variable whenever possible.",
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
    api_key = os.environ.get(config.api_key_env, "")
    base_url = config.base_url or (os.environ.get(config.base_url_env) if config.base_url_env else None)
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


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def _redacted(config: LLMDecisionConfig) -> bool:
    return "XX" in {config.provider, config.model, config.api_key_env}


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
    "LLMDecisionAdvisor",
    "LLMDecisionAdvisorResult",
    "LLMDoctorDraft",
    "LLMEvidenceRequest",
    "LLMProposalBundle",
    "LLMRejectedAction",
]
