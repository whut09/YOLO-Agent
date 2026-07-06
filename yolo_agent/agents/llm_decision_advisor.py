"""Default LLM proposal generator for loop policy planning."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

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
    proposals: list[CandidatePolicy] = Field(default_factory=list)
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
        if not config.can_generate_proposals:
            return _result(config, "skipped", warnings=["llm_decision_config_disabled"])
        if _redacted(config):
            return _result(config, "skipped", warnings=["llm_decision_config_redacted"])
        api_key = os.environ.get(config.api_key_env)
        if config.require_api_key and not api_key:
            return _result(config, "skipped", warnings=[f"missing_api_key_env:{config.api_key_env}"])

        messages = _messages(config, task_spec, diagnosis_report, inherited_context or {})
        try:
            raw_text = self.transport(config, messages)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return _result(config, "failed", warnings=[f"llm_call_failed:{exc}"])

        proposals, warnings = _parse_proposals(raw_text)
        return LLMDecisionAdvisorResult(
            status="used" if proposals else "failed",
            provider=config.provider,
            model=config.model,
            model_alias=config.model_alias,
            proposals=proposals,
            warnings=warnings if proposals else [*warnings, "llm_returned_no_valid_policy_proposals"],
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
                "doctor_report_draft": "optional object",
                "evidence_requests": "optional list",
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


def _parse_proposals(raw_text: str) -> tuple[list[CandidatePolicy], list[str]]:
    warnings: list[str] = []
    try:
        payload = json.loads(_strip_json_fence(raw_text))
    except json.JSONDecodeError as exc:
        return [], [f"llm_json_parse_failed:{exc}"]
    raw_policies = payload.get("candidate_policies", payload.get("policies", [])) if isinstance(payload, dict) else payload
    if not isinstance(raw_policies, list):
        return [], ["llm_candidate_policies_not_list"]
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
    return proposals, warnings


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
) -> LLMDecisionAdvisorResult:
    return LLMDecisionAdvisorResult(
        status=status,
        provider=config.provider,
        model=config.model,
        model_alias=config.model_alias,
        warnings=warnings or [],
    )


__all__ = ["LLMDecisionAdvisor", "LLMDecisionAdvisorResult"]
